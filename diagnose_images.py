"""
图片检测诊断脚本 -- 逐阶段测试图片提取->存储->匹配流程
"""
import sys, os, json, io, logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
sys.path.insert(0, '.')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import fitz
from PIL import Image
import imagehash

from config import DetectionConfig
from extraction.pdf_extractor import PyMuPDFExtractor
from extraction.feature_cache import DocumentCache
from image_analysis.image_hasher import ImageHasher, ImageSignature
from image_analysis.image_matcher import ImageMatcher

INPUT_DIR = './input/'
CACHE_DIR = config = DetectionConfig().CACHE_DIR

def step1_check_pdf_images():
    """Step 1: Open each PDF and check what images PyMuPDF finds."""
    print("=" * 60)
    print("STEP 1: 检查 PDF 中的图片")
    print("=" * 60)
    pdfs = sorted(glob('input/*.pdf'))
    if not pdfs:
        pdfs = sorted(glob('./input/*.pdf'))
    if not pdfs:
        print("[FAIL] 未找到 PDF 文件!")
        return []

    for pdf_path in pdfs:
        print(f"\n[FILE] {os.path.basename(pdf_path)}:")
        doc = fitz.open(pdf_path)
        print(f"   总页数: {doc.page_count}")

        total_images = 0
        total_image_info = 0
        for page_num in range(doc.page_count):
            page = doc[page_num]
            img_list = page.get_images(full=True)
            total_images += len(img_list)
            img_info = page.get_image_info()
            total_image_info += len(img_info)

            if img_list:
                for img in img_list:
                    xref = img[0]
                    w, h = img[2], img[3]
                    base = doc.extract_image(xref)
                    img_bytes = len(base["image"]) if base and base.get("image") else 0
                    print(f"   第{page_num}页: get_images xref={xref}, size={w}x{h}pt, bytes={img_bytes}")

            if img_info:
                for info in img_info[:5]:
                    bbox = info.get('bbox', (0,0,0,0))
                    iw, ih = info.get('width', 0), info.get('height', 0)
                    print(f"   第{page_num}页: get_image_info bbox={bbox}, size={iw}x{ih}")

        print(f"   汇总: get_images={total_images}, get_image_info={total_image_info}")
        doc.close()

    return pdfs


def step2_test_extract_embedded(pdfs):
    """Step 2: Run the actual _extract_embedded_images and check output."""
    print("\n" + "=" * 60)
    print("STEP 2: 运行 _extract_embedded_images")
    print("=" * 60)
    config = DetectionConfig()
    extractor = PyMuPDFExtractor(config)

    all_hashes = {}
    for pdf_path in pdfs:
        doc = fitz.open(pdf_path)
        fname = os.path.basename(pdf_path)
        hashes = extractor._extract_embedded_images(doc, 0, doc.page_count)
        all_hashes[fname] = hashes
        print(f"\n[FILE] {fname}: {len(hashes)} 个图片哈希")
        for h in hashes:
            print(f"   {h[:20]}...")
        doc.close()

    if len(all_hashes) >= 2:
        print("\n--- 文档间哈希交集 ---")
        names = list(all_hashes.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                common = set(all_hashes[names[i]]) & set(all_hashes[names[j]])
                print(f"{names[i]} vs {names[j]}: {len(common)} 个相同哈希")
                for h in common:
                    print(f"   [MATCH] {h}")

    return all_hashes


def step3_check_database():
    """Step 3: Check what's in the database."""
    print("\n" + "=" * 60)
    print("STEP 3: 检查数据库中的 image_hashes")
    print("=" * 60)
    config = DetectionConfig()
    cache = DocumentCache(config.CACHE_DIR, config)

    docs = cache.load_all_documents()
    if not docs:
        print("[FAIL] 数据库中没有文档! (可能未运行过检测)")
        cache.close()
        return {}

    for doc in docs:
        print(f"\n[FILE] {doc.filename}:")
        print(f"   image_hashes: {len(doc.image_hashes)} 个")
        for h in doc.image_hashes:
            print(f"     {h[:20]}...")
        print(f"   doc_minhash: {'YES' if doc.doc_minhash else 'NO'}")
        print(f"   page_count: {doc.page_count}")

    if len(docs) >= 2:
        print("\n--- 数据库中文档间共同图片哈希 ---")
        for i in range(len(docs)):
            for j in range(i+1, len(docs)):
                common = set(docs[i].image_hashes) & set(docs[j].image_hashes)
                print(f"{docs[i].filename} vs {docs[j].filename}: {len(common)} 个相同")
                for h in common:
                    print(f"   [MATCH] {h}")

    cursor = cache.conn.cursor()
    cursor.execute("SELECT pair_id, image_match_count, evidence_json FROM pairwise_results ORDER BY pair_id")
    rows = cursor.fetchall()
    if rows:
        print("\n--- 已存储的配对结果 ---")
        for row in rows:
            pair_id, img_cnt, ev_json = row
            ev = json.loads(ev_json or '{}')
            exact = ev.get('image_exact_count', 0)
            near = ev.get('image_near_identical_count', 0)
            typos = ev.get('shared_typo_count', 0)
            print(f"  {pair_id}: image_match_count={img_cnt}, exact={exact}, near={near}, typos={typos}")
            print(f"    risk_factors: {ev.get('image_risk_factors', [])}")

    cache.close()
    return {d.filename: d for d in docs}


def step4_test_matcher_directly(pdfs):
    """Step 4: Directly test the image matcher on extracted hashes."""
    print("\n" + "=" * 60)
    print("STEP 4: 直接测试 ImageMatcher")
    print("=" * 60)

    config = DetectionConfig()
    extractor = PyMuPDFExtractor(config)
    matcher = ImageMatcher()

    all_hashes = {}
    for pdf_path in pdfs:
        doc = fitz.open(pdf_path)
        fname = os.path.basename(pdf_path)
        hashes = extractor._extract_embedded_images(doc, 0, doc.page_count)
        all_hashes[fname] = hashes
        doc.close()

    if len(all_hashes) >= 2:
        names = list(all_hashes.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                ha, hb = all_hashes[names[i]], all_hashes[names[j]]
                print(f"\n{names[i]} ({len(ha)} hashes) vs {names[j]} ({len(hb)} hashes)")

                common = set(ha) & set(hb)
                print(f"  4a 直接交集: {len(common)} 个相同")

                result = matcher.analyze(hashes_a=ha, hashes_b=hb)
                print(f"  4b L1结果: exact={result.exact_image_count}, near={result.near_identical_count}, similar={result.similar_image_count}")
                print(f"    风险因素: {result.image_risk_factors}")
                for v in result.image_verdicts:
                    print(f"    判决: phash_dist={v.phash_dist}, dhash_dist={v.dhash_dist}, confidence={v.confidence:.3f}")
                    print(f"      理由: {v.reasons}")
                    if v.sig_a.phash:
                        print(f"      sig_a: phash={v.sig_a.phash[:16]}...")
                    if v.sig_b.phash:
                        print(f"      sig_b: phash={v.sig_b.phash[:16]}...")


if __name__ == '__main__':
    from glob import glob

    pdfs = step1_check_pdf_images()
    if not pdfs:
        sys.exit(1)

    hashes = step2_test_extract_embedded(pdfs)

    print("\n\n" + "=" * 60)
    print("检查数据库（若已运行过检测）")
    print("=" * 60)
    db_docs = step3_check_database()

    step4_test_matcher_directly(pdfs)

    print("\n诊断完成!")
