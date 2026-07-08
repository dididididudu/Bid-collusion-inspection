"""
快速诊断脚本 — 测试 PDF 文本提取和检测管线的每个阶段
用法: python diagnose.py
"""
import sys, os, json, tempfile, shutil

sys.path.insert(0, '.')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("=" * 60)
print("[1/5] 生成测试 PDF...")
from test_api import create_simple_pdf

tmpdir = tempfile.mkdtemp()
pdf_path = create_simple_pdf(
    tmpdir, "诊断测试文件",
    ["## 第一章 公司概况",
     "测试公司成立于 2010 年，注册资本 5000 万元。",
     "## 第二章 服务承诺",
     "本公司承诺完全响应招标文件的所有技术要求。",
     "## 第三章 联系方式",
     "联系人：张三",
     "联系电话：13800138000",
     "电子邮箱：zhangsan@test.com"],
    file_id="",
    author="张三",
    contact=("张三", "13800138000", "zhangsan@test.com"),
)
print(f"  PDF 已生成")

# 2. 测试文本提取和 _split_paragraphs
print("\n[2/5] 测试 _split_paragraphs 分段结果...")
import fitz
from extraction.pdf_extractor import PyMuPDFExtractor
from config import DetectionConfig
config = DetectionConfig()
extractor = PyMuPDFExtractor(config)

doc = fitz.open(pdf_path)
page_text = doc[0].get_text("text")
doc.close()
print(f"  原始文本长度: {len(page_text)}")
paras = extractor._split_paragraphs(page_text)
print(f"  分段数: {len(paras)}")
for i, p in enumerate(paras):
    print(f"    段[{i}]: len={len(p)}, text={p[:100]}")

# 3. 测试联系人提取（直接从文本）
print("\n[3/5] 测试联系人提取（直接从文本）...")
from extraction.contact_extractor import extract_contacts_from_text
fp = extract_contacts_from_text(page_text)
print(f"  公司名: {fp.company_names}")
print(f"  联系人: {fp.contact_names}")
print(f"  手机号: {fp.mobile_phones}")
print(f"  邮箱: {fp.emails}")

# 4. 测试联系人提取（从 SQLite 段落表）
print("\n[4/5] 测试联系人提取（从 SQLite 段落表）...")
from extraction.feature_cache import DocumentCache
from extraction.text_processor import ChunkedTextProcessor

cache_dir = os.path.join(tmpdir, "cache")
cache = DocumentCache(cache_dir, config)
text_processor = ChunkedTextProcessor(config)

# 提取并存储
pdf1 = create_simple_pdf(tmpdir, "公司A",
    ["## 第一章", "公司A, 成立于 2010 年。",
     "## 联系方式", "联系人：张三", "电话：13800138000"],
    author="张三")

doc_id = extractor._generate_doc_id(pdf1)
metadata, page_count, is_scanned = extractor.extract_metadata(pdf1)
filename = os.path.basename(pdf1)
file_size = os.path.getsize(pdf1)

for cr in extractor.extract_chunks(pdf1, 50, 0):
    cache.store_chunk(cr)

feature = text_processor.aggregate_chunks(
    doc_id=doc_id, filename=filename, file_size=file_size,
    chunks=[cr], metadata=metadata,
    is_scanned=False, page_count=page_count,
)
cache.store_document(feature)

# 查段落表
cur = cache.conn.execute("SELECT COUNT(*), LENGTH(text) FROM paragraphs WHERE doc_id = ?", (doc_id,))
cnt, txtlen = cur.fetchone()
cur2 = cache.conn.execute("SELECT text FROM paragraphs WHERE doc_id = ?", (doc_id,))
row = cur2.fetchone()
if row:
    print(f"  段落表: {cnt}行, text长度={txtlen}")
    print(f"  全文: {row[0][:120]}")
else:
    print("  段落表: 无数据!")

from extraction.contact_extractor import extract_contacts_from_sqlite
fp2 = extract_contacts_from_sqlite(doc_id, cache)
print(f"  联系人: {fp2.contact_names}")
print(f"  手机号: {fp2.mobile_phones}")

cache.close()
shutil.rmtree(tmpdir)
print("\n诊断完成!")
