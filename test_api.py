"""
API 测试脚本 — 调用检测 API

用法:
  1. 先启动 API 服务:  python api_server.py
  2a. 使用指定 PDF 文件:  python test_api.py --input-dir ./bids/
  2b. 使用生成的测试 PDF: python test_api.py

脚本自动:
  - 调用 /api/detect 上传并启动检测
  - 轮询等待结果（含增量结果展示）
  - 下载 PDF 报告到本地
"""

import os
import sys
import json
import time
import uuid
import shutil
import argparse
import urllib.request
import urllib.error
from io import BytesIO

# ================================================================
# 配置
# ================================================================
API_BASE = "http://localhost:8000"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output")

# ================================================================
# 第一步：生成测试 PDF 文件
# ================================================================

def create_test_pdfs(output_dir: str):
    """生成 4 份测试标书 PDF，含不同层级的重叠内容"""

    # 公用的重叠段落
    shared_paragraphs = [
        "本公司承诺完全响应招标文件的所有技术要求，提供不少于三年的免费质保期服务。"
        "在质保期内，投标人应提供 7×24 小时技术支持服务，接到故障报修后 2 小时内响应，"
        "4 小时内到达现场，8 小时内解决问题。",

        "项目管理团队由项目经理、技术负责人、质量管理员、安全管理员组成。"
        "项目经理须持有高级工程师职称证书，并具有不少于 10 年的同类项目管理经验。",

        "验收标准应符合招标文件第四章相关规定，包括功能验收、性能验收、安全验收三个阶段。"
        "每个阶段验收通过后方可进入下一阶段。",
    ]

    company_a_specific = [
        "公司 A 成立于 2010 年，注册资本 5000 万元，拥有 ISO9001 质量管理体系认证。",
        "公司 A 近三年完成同类项目 15 个，项目总金额超过 2 亿元。",
    ]

    company_b_specific = [
        "公司 B 成立于 2012 年，注册资本 8000 万元，拥有 ISO9001 质量管理体系认证、"
        "ISO14001 环境管理体系认证。",
        "公司 B 近三年完成同类项目 20 个，项目总金额超过 3 亿元。",
    ]

    company_c_specific = [
        "公司 C 成立于 2015 年，注册资本 3000 万元，拥有 ISO9001 质量管理体系认证。",
        "项目管理团队由项目经理、技术负责人、质量管理员、安全管理员组成。"
        "项目经理须持有高级工程师职称证书，并具有不少于 10 年的同类项目管理经验。",
        "公司 C 近三年完成同类项目 18 个，项目总金额超过 2.5 亿元。",
    ]

    company_d_specific = [
        "公司 D 成立于 2018 年，注册资本 2000 万元，专注于政务信息化建设。",
        "公司 D 在智慧政务领域拥有多项自主知识产权软件产品。",
    ]

    file_ids = {
        "公司A_技术标书": "ABCD1234-EF56-7890-GH12-IJ345678KL01",
        "公司B_技术标书": "MNOP5678-QR90-1234-ST56-UV789012WX34",
        "公司C_技术标书": "ABCD1234-EF56-7890-GH12-IJ345678KL01",
        "公司D_技术标书": "YZAB9012-CD34-5678-EF90-GH123456IJ78",
    }

    authors = {
        "公司A_技术标书": "张三",
        "公司B_技术标书": "张三",
        "公司C_技术标书": "李四",
        "公司D_技术标书": "王五",
    }

    contacts = {
        "公司A_技术标书": ("张三", "13800138000", "zhangsan@a-company.com"),
        "公司B_技术标书": ("李四", "13800138000", "lisi@b-company.com"),
        "公司C_技术标书": ("王五", "13900139000", "wangwu@c-company.com"),
        "公司D_技术标书": ("赵六", "13700137000", "zhaoliu@d-company.com"),
    }

    pdf_files = []
    for name in ["公司A_技术标书", "公司B_技术标书", "公司C_技术标书", "公司D_技术标书"]:
        paragraphs = []

        paragraphs.append(f"# {name}\n\n项目名称：XX 市智慧政务平台建设项目\n"
                          f"投标单位：{name.replace('_', '')}\n"
                          f"投标日期：2026 年 7 月\n")

        paragraphs.append("## 第一章 公司概况\n")
        if "公司A" in name:
            paragraphs.extend(company_a_specific)
        elif "公司B" in name:
            paragraphs.extend(company_b_specific)
        elif "公司C" in name:
            paragraphs.extend(company_c_specific)
        elif "公司D" in name:
            paragraphs.extend(company_d_specific)

        paragraphs.append("## 第二章 服务承诺\n")
        paragraphs.append(shared_paragraphs[0])

        paragraphs.append("## 第三章 项目管理\n")
        paragraphs.append(shared_paragraphs[1])

        paragraphs.append("## 第四章 验收标准\n")
        paragraphs.append(shared_paragraphs[2])

        import random
        base_price = random.randint(800, 1200)
        paragraphs.append(f"\n## 第五章 报价\n\n投标总价：{base_price} 万元人民币。\n")

        pdf_path = create_simple_pdf(output_dir, name, paragraphs,
                                      file_id=file_ids[name],
                                      author=authors[name],
                                      contact=contacts[name])
        pdf_files.append(pdf_path)
        print(f"  生成测试文件: {pdf_path}")

    return pdf_files


def create_simple_pdf(output_dir: str, title: str, paragraphs: list,
                      file_id="", author="", contact=("", "", "")) -> str:
    """用 ReportLab 生成一份含中文内容的测试 PDF"""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm, cm
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

    font_dir = "C:/Windows/Fonts"
    try:
        pdfmetrics.registerFont(TTFont('YaHei', os.path.join(font_dir, 'msyh.ttc')))
        pdfmetrics.registerFont(TTFont('YaHei-Bold', os.path.join(font_dir, 'msyhbd.ttc')))
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)
    safe_name = title.replace('/', '_').replace('\\', '_').replace(' ', '_')
    pdf_path = os.path.join(output_dir, f"{safe_name}.pdf")

    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            topMargin=2*cm, bottomMargin=2*cm,
                            leftMargin=2.5*cm, rightMargin=2.5*cm)

    style_normal = ParagraphStyle('Normal', fontName='YaHei', fontSize=10,
                                  leading=18, spaceAfter=4*mm)
    style_heading = ParagraphStyle('Heading', fontName='YaHei-Bold', fontSize=14,
                                    spaceBefore=6*mm, spaceAfter=3*mm)
    story = []

    for para in paragraphs:
        if para.startswith('# '):
            story.append(Paragraph(para[2:], style_heading))
        elif para.startswith('## '):
            story.append(Paragraph(para[3:], ParagraphStyle(
                'H2', fontName='YaHei-Bold', fontSize=12,
                spaceBefore=4*mm, spaceAfter=2*mm)))
        else:
            lines = para.strip().split('\n')
            for line in lines:
                if line.strip():
                    story.append(Paragraph(line.strip(), style_normal))

    if contact[0] or contact[1] or contact[2]:
        story.append(Paragraph("联系方式", style_heading))
        if contact[0]:
            story.append(Paragraph(f"联系人：{contact[0]}", style_normal))
        if contact[1]:
            story.append(Paragraph(f"联系电话：{contact[1]}", style_normal))
        if contact[2]:
            story.append(Paragraph(f"电子邮箱：{contact[2]}", style_normal))

    doc.build(story)

    import fitz
    try:
        pdf_doc = fitz.open(pdf_path)
        pdf_doc.set_metadata({
            'title': title,
            'author': author,
        })
        pdf_doc.save(pdf_path, incremental=True, encryption=0)
        pdf_doc.close()
    except Exception:
        pass

    return pdf_path


# ================================================================
# 第二步：调用 API 进行检测
# ================================================================

def call_api(method: str, path: str, data=None, files=None,
             multipart_data=None):
    """调用 FastAPI 接口，返回 (status, data)"""
    import http.client
    import mimetypes

    url = f"{API_BASE}{path}"

    if files:
        boundary = uuid.uuid4().hex
        body = BytesIO()

        if data:
            for key, value in data.items():
                body.write(f"--{boundary}\r\n".encode())
                body.write(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
                body.write(f"{value}\r\n".encode())

        for field_name, file_path in files:
            with open(file_path, 'rb') as f:
                file_content = f.read()
            filename = os.path.basename(file_path)
            body.write(f"--{boundary}\r\n".encode())
            body.write(f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode())
            body.write(f"Content-Type: application/pdf\r\n\r\n".encode())
            body.write(file_content)
            body.write(b"\r\n")

        body.write(f"--{boundary}--\r\n".encode())
        body_data = body.getvalue()
        content_type = f"multipart/form-data; boundary={boundary}"

        req = urllib.request.Request(url, data=body_data, method=method)
        req.add_header("Content-Type", content_type)
    else:
        req = urllib.request.Request(url, method=method)
        if data:
            req.add_header("Content-Type", "application/json")
            req.data = json.dumps(data).encode()

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = resp.read()
            content_type = resp.headers.get('Content-Type', '')
            if 'application/json' in content_type:
                return resp.status, json.loads(body.decode())
            return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body.decode())
        except Exception:
            return e.code, body.decode()
    except urllib.error.URLError as e:
        print(f"  [错误] 无法连接到 API 服务 ({API_BASE})，请确保已启动 python api_server.py")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='投标文件串标围标检测 — API 测试脚本')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='指定本地 PDF 文件目录（不指定则自动生成测试文件）')
    args = parser.parse_args()

    print("=" * 60)
    print("投标文件串标围标检测 — API 测试脚本")
    print("=" * 60)

    global OUTPUT_DIR
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: 检查 API 健康状态
    print("\n[1/5] 检查 API 服务状态...")
    status, data = call_api("GET", "/api/health")
    print(f"  API 状态: {data.get('status', 'unknown')}")

    # Step 2: 获取可用维度
    print("\n[2/5] 获取可用检测维度...")
    status, data = call_api("GET", "/api/dimensions")
    dims = data.get('dimensions', [])
    for d in dims:
        print(f"  - {d['name']} ({d['id']})")

    # Step 3: 准备 PDF 文件
    if args.input_dir:
        print(f"\n[3/5] 从目录加载 PDF 文件: {args.input_dir}")
        if not os.path.isdir(args.input_dir):
            print(f"  [错误] 目录不存在: {args.input_dir}")
            return
        pdf_files = sorted([
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.lower().endswith('.pdf')
        ])
        if not pdf_files:
            print(f"  [错误] 目录中没有 PDF 文件: {args.input_dir}")
            return
        if len(pdf_files) < 2:
            print(f"  [错误] 至少需要 2 个 PDF 文件，当前 {len(pdf_files)} 个")
            return
        print(f"  共加载 {len(pdf_files)} 个 PDF 文件")
        for f in pdf_files:
            print(f"    - {os.path.basename(f)} ({os.path.getsize(f) / 1024:.0f} KB)")
    else:
        print("\n[3/5] 生成测试 PDF 标书文件...")
        test_pdf_dir = os.path.join(OUTPUT_DIR, "generated_pdfs")
        pdf_files = create_test_pdfs(test_pdf_dir)
        print(f"  共生成 {len(pdf_files)} 个测试文件")

    # Step 4: 提交检测任务
    print("\n[4/5] 提交检测任务...")
    files_for_api = [("files", pf) for pf in pdf_files]
    data = {"content_similarity": "true"}
    status, result = call_api("POST", "/api/detect", data=data, files=files_for_api)

    task_id = result.get("task_id", "")
    print(f"  任务已提交: {task_id}")
    print(f"  状态: {result.get('status', '')}")

    # Step 5: 轮询等待结果
    print("\n[5/5] 轮询等待检测结果...")
    prev_count = 0
    while True:
        time.sleep(2)
        status, result = call_api("GET", f"/api/detect/{task_id}")

        task_status = result.get("status", "")
        progress = result.get("progress", {})

        partial = result.get("partial_results", [])
        if partial and len(partial) > prev_count:
            for p in partial[prev_count:]:
                fn_a = p.get('filename_a', '?')
                fn_b = p.get('filename_b', '?')
                sim = p.get('text_similarity', 0)
                mc = p.get('match_count', 0)
                print(f"  [增量] {fn_a} ↔ {fn_b}  相似度={sim:.4f}  匹配={mc}段")
            prev_count = len(partial)

        phase = progress.get('phase', '')
        cur = progress.get('current', 0)
        total = progress.get('total', 0)
        if total > 0:
            pct = cur / total * 100
            bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
            print(f"  进度: [{bar}] {cur}/{total}  {phase}", end='\r')

        if task_status == "completed":
            print(f"\n  {'=' * 50}")
            print(f"  检测完成!")
            print(f"  总文件: {result.get('result', {}).get('total_files', '?')}")
            print(f"  可疑对: {result.get('result', {}).get('suspicious_pairs', '?')}")
            print(f"  耗时: {result.get('elapsed_seconds', '?')} 秒")
            print(f"  {'=' * 50}")
            break
        elif task_status == "failed":
            print(f"\n  [错误] 检测失败: {result.get('error', '未知错误')}")
            return
        else:
            print(f"  状态: {task_status}  {phase}  ({cur}/{total})", end='\r')

    # 下载 PDF 报告
    print("\n  正在下载 PDF 报告...")
    report_url = result.get("report_url", f"/api/detect/{task_id}/report")
    status, pdf_data = call_api("GET", report_url)

    pdf_path = os.path.join(OUTPUT_DIR, "detection_report.pdf")
    with open(pdf_path, 'wb') as f:
        f.write(pdf_data if isinstance(pdf_data, bytes) else pdf_data.encode())
    print(f"  PDF 报告已保存: {pdf_path}")

    json_result = result.get("result", {})
    json_path = os.path.join(OUTPUT_DIR, "detection_result.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, indent=2, ensure_ascii=False)
    print(f"  JSON 结果已保存: {json_path}")

    dims_hit = json_result.get("dimensions", {})
    if dims_hit:
        print(f"\n  维度命中情况:")
        for dim_key, dim_info in dims_hit.items():
            hit_str = "hit" if dim_info.get("hit") else "miss"
            enabled_str = "on" if dim_info.get("enabled") else "off"
            print(f"    {dim_key}: {hit_str} ({enabled_str})")

    print(f"\n  {'=' * 50}")
    print(f"  报告文件位置: {OUTPUT_DIR}")
    print(f"  PDF 报告: {pdf_path}")
    print(f"  {'=' * 50}")
    print("\n测试完成！")


if __name__ == "__main__":
    main()
