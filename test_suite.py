"""全功能自动化测试套件 — 投标文件串标围标检测"""
import sys, os, json, tempfile, shutil, time, traceback, random
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

TMPDIR = tempfile.mkdtemp(prefix='bid_test_')
RESULTS = []
TIMING = []
ERRORS = []
TOTAL_CNT = defaultdict(int)
PASS_CNT = defaultdict(int)

# ================================================================
# PDF 生成工具
# ================================================================
def _make_pdf(path, pages, meta=None):
    import fitz

    doc = fitz.open()
    for txt in pages:
        page = doc.new_page(width=595, height=842)  # A4 points
        y = 72
        for raw_line in txt.strip().split('\n'):
            line = raw_line.strip()
            if not line:
                y += 12
                continue
            is_heading = line.startswith('# ')
            if is_heading:
                line = line[2:]
            font_size = 14 if is_heading else 10
            for start in range(0, len(line), 42):
                page.insert_text(
                    (72, y), line[start:start + 42],
                    fontsize=font_size,
                    fontname="china-s",
                )
                y += 22 if is_heading else 16
                if y > 780:
                    page = doc.new_page(width=595, height=842)
                    y = 72
    if meta:
        doc.set_metadata({k: v for k, v in meta.items() if v})
    doc.save(path)
    doc.close()

def make_pdf_3page(path, company, contact, cv, author='', creator='', producer=''):
    intro = cv.get('intro','公司简介。')
    about = cv.get('about','业务介绍。')
    qual = cv.get('qualification','资质认证。')
    tech = cv.get('tech_approach','技术方案。')
    detail = cv.get('tech_detail','技术细节。')
    qa = cv.get('quality_assurance','质量保证。')
    ccl = cv.get('credit_code_line','')
    extra = cv.get('extra','')
    pages = [
        f"# {company} 投标文件\n\n项目名称：XX智慧政务平台\n\n{intro}\n\n{about}\n\n{qual}",
        f"# 技术方案\n\n{tech}\n\n{detail}\n\n{qa}",
        f"# 联系方式\n\n联系人：{contact.get('name','')}\n联系电话：{contact.get('phone','')}\n电子邮箱：{contact.get('email','')}\n\n# 公司信息\n\n公司名称：{company}\n{ccl}\n{extra}"
    ]
    return _make_pdf(path, pages, {'author':author,'creator':creator,'producer':producer} or None)


# ================================================================
# 测试数据
# ================================================================
TEXT_SIM_CASES = [
    ("标书模板完全相同",
     ("A公司",{"name":"张","phone":"13800000001","email":"a@a.com"},
      {"intro":"我公司成立于2010年注册资本5000万元。","about":"专注于政务信息化建设十五年。",
       "qualification":"拥有ISO9001和CMMI5等资质认证。",
       "tech_approach":"采用微服务架构支持高并发低延迟。","tech_detail":"系统基于SpringCloud框架。",
       "quality_assurance":"提供7x24小时技术支持2小时响应。"}),
     ("B公司",{"name":"李","phone":"13900000002","email":"b@b.com"},
      {"intro":"我公司成立于2010年注册资本5000万元。","about":"专注于政务信息化建设十五年。",
       "qualification":"拥有ISO9001和CMMI5等资质认证。",
       "tech_approach":"采用微服务架构支持高并发低延迟。","tech_detail":"系统基于SpringCloud框架。",
       "quality_assurance":"提供7x24小时技术支持2小时响应。"}),
     True, "完全相同内容应匹配"),

    ("部分段落重叠",
     ("C公司",{"name":"王","phone":"13700000003","email":"c@c.com"},
      {"intro":"我公司专注于政务信息化领域具有丰富经验。","about":"已完成多个大型项目。",
       "qualification":"拥有ISO9001和系统集成资质。",
       "tech_approach":"采用微服务架构支持高并发低延迟。","tech_detail":"系统基于SpringCloud框架构建。",
       "quality_assurance":"提供5x8小时技术支持服务。"}),
     ("D公司",{"name":"赵","phone":"13600000004","email":"d@d.com"},
      {"intro":"我们专注于大数据分析领域。","about":"核心产品是数据分析平台。",
       "qualification":"拥有ISO9001资质。",
       "tech_approach":"采用微服务架构支持高并发低延迟。","tech_detail":"系统基于SpringCloud框架构建。",
       "quality_assurance":"提供7x24小时技术支持服务。"}),
     True, "部分重叠应匹配"),

    ("完全不同行业不应匹配",
     ("E公司",{"name":"钱","phone":"13500000005","email":"e@e.com"},
      {"intro":"本公司专注于人工智能领域核心产品为智能客服机器人。","about":"拥有多项AI软件著作权。",
       "qualification":"已通过ISO9001认证。",
       "tech_approach":"采用深度学习框架PyTorch训练模型。","tech_detail":"模型部署在GPU集群上推理延迟低于50毫秒。",
       "quality_assurance":"提供7x24小时在线文档和技术支持。"}),
     ("F公司",{"name":"孙","phone":"13400000006","email":"f@f.com"},
      {"intro":"本公司专注于生物医药研发核心产品为创新药物。","about":"拥有多项药品发明专利。",
       "qualification":"已通过GMP药品质量管理认证。",
       "tech_approach":"采用基因编辑技术CRISPRCas9进行药物开发。","tech_detail":"实验数据表明药效提升百分之三十。",
       "quality_assurance":"严格按照GLP药物非临床研究规范执行实验。"}),
     False, "完全不同行业不应匹配"),

    ("共用招标模板",
     ("G公司",{"name":"周","phone":"13300000007","email":"g@g.com"},
      {"intro":"我公司成立于2015年。具有丰富的行业经验。","about":"拥有相关资质认证。",
       "qualification":"详见资质证书附件。",
       "tech_approach":"技术方案详见投标文件。","tech_detail":"具体参数见技术规格书。",
       "quality_assurance":"提供售后服务。"}),
     ("H公司",{"name":"吴","phone":"13200000008","email":"h@h.com"},
      {"intro":"我公司成立于2016年。具有丰富的行业经验。","about":"拥有相关资质认证。",
       "qualification":"详见资质证书附件。",
       "tech_approach":"技术方案详见投标文件。","tech_detail":"具体参数见技术规格书。",
       "quality_assurance":"提供售后服务。"}),
     True, "共有模板语应匹配"),

    ("长文本部分重复",
     ("I公司",{"name":"郑","phone":"13100000009","email":"i@i.com"},
      {"intro":"我公司成立于2012年注册资本1亿元是国家级高新技术企业。多年来深耕智慧城市领域。",
       "about":"拥有员工500余人其中研发人员占比60%以上。通过了CMMI5认证。",
       "qualification":"通过ISO9001/ISO14001/ISO45001三体系认证。",
       "tech_approach":"采用业界领先的云原生技术架构基于Kubernetes容器编排平台实现弹性伸缩。",
       "tech_detail":"前端使用React框架后端使用SpringBoot微服务架构。",
       "quality_assurance":"提供7x24小时技术支持服务2小时响应4小时到场8小时解决问题。"}),
     ("J公司",{"name":"冯","phone":"13000000010","email":"j@j.com"},
      {"intro":"我公司成立于2013年注册资本8000万元是国家级高新技术企业。多年来深耕金融科技领域。",
       "about":"拥有员工300余人其中研发人员占比50%以上。通过了CMMI3认证。",
       "qualification":"通过ISO9001和ISO27001认证。",
       "tech_approach":"采用业界领先的云原生技术架构基于Kubernetes容器编排平台实现弹性伸缩。",
       "tech_detail":"前端使用Vue框架后端使用SpringBoot微服务架构。",
       "quality_assurance":"提供7x24小时技术支持服务2小时响应4小时到场8小时解决问题。"}),
     True, "长文本部分重复应匹配"),

    ("极短内容相同",
     ("超短A",{"name":"甲","phone":"1","email":"a@b"},
      {"intro":"本公司投标。","about":"。","qualification":"。",
       "tech_approach":"。","tech_detail":"。","quality_assurance":"。"}),
     ("超短B",{"name":"乙","phone":"2","email":"b@c"},
      {"intro":"本公司投标。","about":"。","qualification":"。",
       "tech_approach":"。","tech_detail":"。","quality_assurance":"。"}),
     True, "极短但相同应匹配"),

    ("含编码的相似模板",
     ("K公司",{"name":"张","phone":"13800000011","email":"k@k.com"},
      {"intro":"项目编号GCHG2024001项目预算580万元。","about":"合同编号HT2024001。",
       "qualification":"证书编号ZJ2024001。",
       "tech_approach":"标准编号GBT222392019。","tech_detail":"设备型号Huawei2288HV7。",
       "quality_assurance":"报修编号BX2024001。"}),
     ("L公司",{"name":"李","phone":"13900000012","email":"l@l.com"},
      {"intro":"项目编号GCHG2024002项目预算620万元。","about":"合同编号HT2024002。",
       "qualification":"证书编号ZJ2024002。",
       "tech_approach":"标准编号GBT222392019。","tech_detail":"设备型号DellR750xs。",
       "quality_assurance":"报修编号BX2024002。"}),
     True, "含编码模板应匹配"),

    ("中英文混合",
     ("M公司",{"name":"Tom","phone":"861088888888","email":"tom@m.com"},
      {"intro":"We are a leading IT company founded in 2010。我们专注于软件开发。",
       "about":"Our team has 200 engineers。核心团队来自知名互联网企业。",
       "qualification":"拥有CMMILevel5和ISO认证。",
       "tech_approach":"采用Agile开发方法Scrum团队协作。","tech_detail":"技术栈包括JavaPythonGo。",
       "quality_assurance":"提供7x24技术支持。"}),
     ("N公司",{"name":"Jerry","phone":"862199999999","email":"jerry@n.com"},
      {"intro":"We are a leading IT company founded in 2012。我们专注于人工智能。",
       "about":"Our team has 150 engineers。核心团队来自知名AI企业。",
       "qualification":"拥有多项AI专利和软件著作权。",
       "tech_approach":"采用Agile开发方法Scrum团队协作。","tech_detail":"技术栈包括PythonTensorFlow。",
       "quality_assurance":"提供5x8技术支持。"}),
     True, "中英文混合相似应匹配"),

    ("全复用标书模板",
     ("投标人X",{"name":"张","phone":"13800000013","email":"x@x.com"},
      {"intro":"我方完全理解并积极响应本次招标文件所有要求。我公司郑重承诺提供优质产品和服务。",
       "about":"我方保证所提供产品均为原装正品享受厂家标准质保服务。",
       "qualification":"我方具有履行合同所必需的设备和专业技术能力。",
       "tech_approach":"我方承诺按招标文件要求的技术规范实施。","tech_detail":"我方接受招标文件所有商务条款。",
       "quality_assurance":"我方承诺提供不少于三年的免费质保期。"}),
     ("投标人Y",{"name":"李","phone":"13900000014","email":"y@y.com"},
      {"intro":"我方完全理解并积极响应本次招标文件所有要求。我公司郑重承诺提供优质产品和服务。",
       "about":"我方保证所提供产品均为原装正品享受厂家标准质保服务。",
       "qualification":"我方具有履行合同所必需的设备和专业技术能力。",
       "tech_approach":"我方承诺按招标文件要求的技术规范实施。","tech_detail":"我方接受招标文件所有商务条款。",
       "quality_assurance":"我方承诺提供不少于三年的免费质保期。"}),
     True, "全复用模板应匹配"),
]

AUTHOR_CASES = [
    ("相同作者","张三","张三",True), ("不同作者","张三","李四",False),
    ("空作者vs有作者","","张三",False), ("双空作者","","",True),  # PyMuPDF 返回 (anonymous) 相同
    ("三字名相同","张三丰","张三丰",True),
    ("含空格姓名","张 三","张三",False),  # 精确字符串比对
    ("含短横姓名","张-三","张-三",True),
]

EDITOR_CASES = [
    ("相同creator+producer","Word","Microsoft Word","Word","Microsoft Word",True),
    ("仅creator相同","WPS Office","PDFCreator","WPS Office","Acrobat",True),
    ("仅producer相同","Word","Acrobat","LibreOffice","Acrobat",True),
    ("版本号不同","Microsoft Word 2021","Acrobat","Microsoft Word 2019","Acrobat",True),
    ("不同软件","Microsoft Word","Microsoft Word","Adobe Acrobat","Adobe Acrobat",False),
]

CONTACT_CASES = [
    ("相同手机号",("张","13800138000","a@a.com"),("李","13800138000","b@b.com"),True),
    ("相同邮箱",("张","13800000001","same@test.com"),("李","13800000002","same@test.com"),True),
    ("手机+邮箱均不同但页头联系方式相同",("张","13800138000","a@a.com"),("李","13900139000","b@b.com"),True),
    ("同名+不同手机",("张三","13800138000","a@a.com"),("张三","13900139000","b@b.com"),True),
    ("仅手机相同",("","13800000001",""),("","13800000001",""),True),
    ("仅邮箱相同",("","","only@test.com"),("","","only@test.com"),True),
    ("邮箱大小写",("","","Test@Example.com"),("","","test@example.com"),True),
]

COMPANY_CASES = [
    ("完全相同","北京华软科技有限公司","北京华软科技有限公司",True),
    ("不同公司","北京华软科技有限公司","上海智联信息技术有限公司",False),
    ("核心名同地域不同","北京华软科技有限公司","上海华软科技有限公司",False),
    ("含后缀不同","华软科技有限公司","华软科技股份有限公司",False),
    ("简称不匹配","华软科技","华软科技",False),
    ("空vs有","","北京华软科技有限公司",False),
]

CREDIT_CASES = [
    ("完全相同","91110108MA01XXXXX1","91110108MA01XXXXX1",True),
    ("不同代码","91110108MA01XXXXX1","91110108MA01XXXXX2",False),
    ("空代码","","",False),
    ("空vs有","","91110108MA01XXXXX1",False),
]


# ================================================================
# 测试工具
# ================================================================
def _save_result(dim, ok, idx, name, actual, expect, detail):
    TOTAL_CNT[dim] += 1
    if ok: PASS_CNT[dim] += 1
    RESULTS.append((dim, idx, ok, name, actual, expect, detail))
    print(f"    [{['FAIL','PASS'][ok]}] {dim}[{idx}] {name}")

def _except(dim, idx, name, ex, desc):
    TOTAL_CNT[dim] += 1
    tb = traceback.format_exc()
    RESULTS.append((dim, idx, False, name, 'EXCEPTION', 'N/A', f"{desc}: {ex}\n{tb[:300]}"))
    ERRORS.append((dim, idx, name, f"{ex}\n{tb[:200]}"))


# ================================================================
# 1. 文本相似度（含位置随机化测试）
# ================================================================
def test_text_sim():
    print("\n  [文本相似度] ...")
    from config import DetectionConfig
    from extraction.pdf_extractor import PyMuPDFExtractor
    from extraction.text_processor import ChunkedTextProcessor
    from extraction.feature_cache import DocumentCache
    from matching.paragraph_matcher import ParagraphMatcher
    cfg = DetectionConfig(); cfg.CHUNK_PAGE_SIZE=50
    d = os.path.join(TMPDIR,"text_sim"); os.makedirs(d, exist_ok=True)
    ex = PyMuPDFExtractor(cfg)
    tp = ChunkedTextProcessor(cfg)

    # === 位置随机化测试（共享段落出现在文档不同位置）===
    random.seed(42)
    shared_pool = [
        '本公司承诺完全响应招标文件的所有技术要求提供不少于三年的免费质保期服务。',
        '验收标准应符合招标文件第四章相关规定包括功能验收性能验收安全验收三个阶段。',
        '项目经理须持有高级工程师职称证书并具有不少于十年同类项目管理经验。',
        '系统架构采用分布式微服务设计支持水平扩展和故障自动转移。',
    ]
    uni_a = ['公司A成立于2010年注册资本5000万元。','公司A已完成15个同类项目。']
    uni_b = ['公司B成立于2012年注册资本8000万元。','公司B已完成20个同类项目。']
    for trial in range(10):
        TOTAL_CNT['content_similarity'] += 1
        t0=time.time()
        try:
            a_paras=uni_a.copy(); b_paras=uni_b.copy()
            for sp in shared_pool:
                a_paras.insert(random.randint(0,len(a_paras)), sp)
                b_paras.insert(random.randint(0,len(b_paras)), sp)
            p1=os.path.join(d,f"rt{trial}a.pdf"); p2=os.path.join(d,f"rt{trial}b.pdf")
            _make_pdf(p1, ['\n'.join(a_paras[:3]), '\n'.join(a_paras[3:])])
            _make_pdf(p2, ['\n'.join(b_paras[:3]), '\n'.join(b_paras[3:])])
            ca=DocumentCache(os.path.join(d,f"rc{trial}"),cfg); ids=[]
            for p in [p1,p2]:
                meta,pc,sc=ex.extract_metadata(p); did=ex._generate_doc_id(p)
                fn=os.path.basename(p); fs=os.path.getsize(p)
                chs=[]; [ca.store_chunk(c) or chs.append(c) for c in ex.extract_chunks(p,50,0)]
                feat=tp.aggregate_chunks(doc_id=did,filename=fn,file_size=fs,chunks=chs,metadata=meta,is_scanned=False,page_count=pc)
                ca.store_document(feat); ids.append(did)
            da=ca.load_document(ids[0]); db=ca.load_document(ids[1])
            pm=ParagraphMatcher(cfg)
            pa=ca.load_all_paragraphs_full(ids[0]); pb=ca.load_all_paragraphs_full(ids[1])
            ms=pm.match(da,db,ca,para_full_a=pa,para_full_b=pb)
            ok=len(ms)>=len(shared_pool)
            TIMING.append(('content_similarity',20+trial,time.time()-t0))
            _save_result('content_similarity',ok,20+trial,
                f"位置随机化{trial}",ok,True,
                f"共享{len(shared_pool)}段,匹配{len(ms)}段")
        except Exception as e:
            TIMING.append(('content_similarity',20+trial,time.time()-t0))
            _except('content_similarity',20+trial,f"位置随机化{trial}",e,"")

    # === 固定位置的标准测试 ===
    for i,(name,co1,co2,expect,desc) in enumerate(TEXT_SIM_CASES):
        TOTAL_CNT['content_similarity'] += 1
        t0=time.time()
        try:
            p1=os.path.join(d,f"t{i}a.pdf"); p2=os.path.join(d,f"t{i}b.pdf")
            make_pdf_3page(p1,co1[0],co1[1],co1[2],author='T')
            make_pdf_3page(p2,co2[0],co2[1],co2[2],author='T')
            ca=DocumentCache(os.path.join(d,f"c{i}"),cfg); ids=[]
            for p in [p1,p2]:
                meta,pc,sc=ex.extract_metadata(p); did=ex._generate_doc_id(p)
                fn=os.path.basename(p); fs=os.path.getsize(p)
                chs=[]; [ca.store_chunk(c) or chs.append(c) for c in ex.extract_chunks(p,50,0)]
                feat=tp.aggregate_chunks(doc_id=did,filename=fn,file_size=fs,chunks=chs,metadata=meta,is_scanned=False,page_count=pc)
                ca.store_document(feat); ids.append(did)
            da=ca.load_document(ids[0]); db=ca.load_document(ids[1])
            pm=ParagraphMatcher(cfg)
            if not da.doc_minhash or not db.doc_minhash:
                ok=False; dt="无minhash"
            else:
                pa=ca.load_all_paragraphs_full(ids[0]); pb=ca.load_all_paragraphs_full(ids[1])
                ms=pm.match(da,db,ca,para_full_a=pa,para_full_b=pb)
                ok=len(ms)>0; dt=f"{len(ms)}对匹配最高{max((m.get('similarity',0)for m in ms),default=0):.3f}"
            ca.close()
            TIMING.append(('content_similarity',i,time.time()-t0))
            _save_result('content_similarity',ok==expect,i,name,ok,expect,f"{desc}|{dt}")
        except Exception as e:
            TIMING.append(('content_similarity',i,time.time()-t0))
            _except('content_similarity',i,name,e,desc)

# ================================================================
# 其他维度测试
# ================================================================
def test_file_id():
    print("\n  [文件码雷同] ...")
    from config import DetectionConfig
    from extraction.pdf_extractor import PyMuPDFExtractor
    cfg = DetectionConfig(); ex = PyMuPDFExtractor(cfg)
    d = os.path.join(TMPDIR,"fid"); os.makedirs(d,exist_ok=True)
    bc={"intro":"测试文档内容。","about":"测试。","qualification":"。",
        "tech_approach":"。","tech_detail":"。","quality_assurance":"。"}
    bp={"name":"","phone":"","email":""}
    a=os.path.join(d,"f0.pdf"); b=os.path.join(d,"f0b.pdf")
    make_pdf_3page(a,"F0",bp,bc); shutil.copy2(a,b)
    m1,_,_=ex.extract_metadata(a); m2,_,_=ex.extract_metadata(b)
    ok=bool(m1.file_id and m1.file_id==m2.file_id)
    _save_result('file_id',ok,0,"同一副本",ok,True,f"fid1={m1.file_id} fid2={m2.file_id}")
    for i in range(7):
        p=os.path.join(d,f"u{i}.pdf"); make_pdf_3page(p,f"U{i}",bp,bc)
        me,_,_=ex.extract_metadata(p)
        ok2=bool(me.file_id)
        _save_result('file_id',ok2,i+1,f"独立文件{i}",ok2,True,f"fid={me.file_id}")

def test_author():
    print("\n  [文档作者雷同] ...")
    from config import DetectionConfig
    from extraction.pdf_extractor import PyMuPDFExtractor
    cfg = DetectionConfig(); ex = PyMuPDFExtractor(cfg)
    d = os.path.join(TMPDIR,"auth"); os.makedirs(d,exist_ok=True)
    bc={"intro":"测试文档。","about":"用于作者检测。","qualification":"。",
        "tech_approach":"。","tech_detail":"。","quality_assurance":"。"}
    bp={"name":"T","phone":"1","email":"t@t.com"}
    for i,(name,a1,a2,expect) in enumerate(AUTHOR_CASES):
        t0=time.time()
        try:
            p1=os.path.join(d,f"a{i}.pdf"); p2=os.path.join(d,f"b{i}.pdf")
            make_pdf_3page(p1,"A"+str(i),bp,bc,author=a1)
            make_pdf_3page(p2,"B"+str(i),bp,bc,author=a2)
            m1,_,_=ex.extract_metadata(p1); m2,_,_=ex.extract_metadata(p2)
            v1=(m1.author or '').strip().lower(); v2=(m2.author or '').strip().lower()
            ok=bool(v1 and v2 and v1==v2)
            TIMING.append(('author',i,time.time()-t0))
            _save_result('author',ok==expect,i,name,ok,expect,f"a1={m1.author} a2={m2.author}")
        except Exception as e:
            TIMING.append(('author',i,time.time()-t0))
            _except('author',i,name,e,"")

def test_editor():
    print("\n  [编辑经办人雷同] ...")
    from config import DetectionConfig
    from extraction.pdf_extractor import PyMuPDFExtractor
    cfg = DetectionConfig(); ex = PyMuPDFExtractor(cfg)
    d = os.path.join(TMPDIR,"edit"); os.makedirs(d,exist_ok=True)
    bc={"intro":"测试。","about":"用于经办人检测。","qualification":"。",
        "tech_approach":"。","tech_detail":"。","quality_assurance":"。"}
    bp={"name":"T","phone":"1","email":"t@t.com"}
    for i,(name,cr1,pr1,cr2,pr2,expect) in enumerate(EDITOR_CASES):
        t0=time.time()
        try:
            p1=os.path.join(d,f"e{i}.pdf"); p2=os.path.join(d,f"f{i}.pdf")
            make_pdf_3page(p1,"E"+str(i),bp,bc,creator=cr1,producer=pr1)
            make_pdf_3page(p2,"F"+str(i),bp,bc,creator=cr2,producer=pr2)
            m1,_,_=ex.extract_metadata(p1); m2,_,_=ex.extract_metadata(p2)
            flds=['creator','producer','software_fingerprint']
            matched=[f for f in flds if ((getattr(m1,f,'')or'').lower().strip()==(getattr(m2,f,'')or'').lower().strip() and getattr(m1,f,'')and getattr(m2,f,''))]
            ok=len(matched)>0
            TIMING.append(('editor',i,time.time()-t0))
            _save_result('editor',ok==expect,i,name,ok,expect,
                f"cr1={m1.creator} pr1={m1.producer} vs cr2={m2.creator} pr2={m2.producer}")
        except Exception as e:
            TIMING.append(('editor',i,time.time()-t0))
            _except('editor',i,name,e,"")

def test_contact():
    print("\n  [联系人雷同] ...")
    from extraction.contact_extractor import extract_contacts_from_text
    d = os.path.join(TMPDIR,"cont"); os.makedirs(d,exist_ok=True)
    bc={"intro":"公司介绍内容。","about":"关于我们。","qualification":"。",
        "tech_approach":"。","tech_detail":"。","quality_assurance":"。"}
    for i,(name,c1,c2,expect) in enumerate(CONTACT_CASES):
        t0=time.time()
        try:
            p1=os.path.join(d,f"c{i}a.pdf"); p2=os.path.join(d,f"c{i}b.pdf")
            make_pdf_3page(p1,f"C{i}A",{"name":c1[0],"phone":c1[1],"email":c1[2]},bc)
            make_pdf_3page(p2,f"C{i}B",{"name":c2[0],"phone":c2[1],"email":c2[2]},bc)
            import fitz
            t1="".join(p.get_text("text")for p in fitz.open(p1))
            t2="".join(p.get_text("text")for p in fitz.open(p2))
            fp1=extract_contacts_from_text(t1); fp2=extract_contacts_from_text(t2)
            s1=set(fp1.mobile_phones+fp1.emails+[n.strip()for n in fp1.contact_names])
            s2=set(fp2.mobile_phones+fp2.emails+[n.strip()for n in fp2.contact_names])
            ok=len(s1&s2)>0
            TIMING.append(('contact',i,time.time()-t0))
            _save_result('contact',ok==expect,i,name,ok,expect,
                f"A:手机={fp1.mobile_phones}邮箱={fp1.emails}姓名={fp1.contact_names}|"
                f"B:手机={fp2.mobile_phones}邮箱={fp2.emails}姓名={fp2.contact_names}")
        except Exception as e:
            TIMING.append(('contact',i,time.time()-t0))
            _except('contact',i,name,e,"")

def test_company():
    print("\n  [公司名雷同] ...")
    from extraction.contact_extractor import extract_contacts_from_text
    d = os.path.join(TMPDIR,"comp"); os.makedirs(d,exist_ok=True)
    for i,(name,co1,co2,expect) in enumerate(COMPANY_CASES):
        t0=time.time()
        try:
            p1=os.path.join(d,f"cp{i}a.pdf"); p2=os.path.join(d,f"cp{i}b.pdf")
            bp={"name":"T","phone":"1","email":"t@t.com"}
            make_pdf_3page(p1,co1,bp,{"intro":"公司简介。","about":"。","qualification":"。",
                "tech_approach":"。","tech_detail":"。","quality_assurance":"。"})
            make_pdf_3page(p2,co2,bp,{"intro":"公司简介。","about":"。","qualification":"。",
                "tech_approach":"。","tech_detail":"。","quality_assurance":"。"})
            import fitz
            t1="".join(p.get_text("text")for p in fitz.open(p1))
            t2="".join(p.get_text("text")for p in fitz.open(p2))
            fp1=extract_contacts_from_text(t1); fp2=extract_contacts_from_text(t2)
            ok=len(set(fp1.company_names)&set(fp2.company_names))>0
            TIMING.append(('company_name',i,time.time()-t0))
            _save_result('company_name',ok==expect,i,name,ok,expect,
                f"A公司名={fp1.company_names} B公司名={fp2.company_names}")
        except Exception as e:
            TIMING.append(('company_name',i,time.time()-t0))
            _except('company_name',i,name,e,"")

def test_credit():
    print("\n  [信用代码雷同] ...")
    from extraction.contact_extractor import extract_contacts_from_text
    d = os.path.join(TMPDIR,"cred"); os.makedirs(d,exist_ok=True)
    bp={"name":"T","phone":"1","email":"t@t.com"}
    for i,(name,cc1,cc2,expect) in enumerate(CREDIT_CASES):
        t0=time.time()
        try:
            p1=os.path.join(d,f"cr{i}a.pdf"); p2=os.path.join(d,f"cr{i}b.pdf")
            bc1={"intro":"","about":"","qualification":"","tech_approach":"","tech_detail":"","quality_assurance":"",
                 "credit_code_line":f"统一社会信用代码：{cc1}"}
            make_pdf_3page(p1,"CR"+str(i),bp,bc1)
            bc2=dict(bc1); bc2["credit_code_line"]=f"统一社会信用代码：{cc2}"
            make_pdf_3page(p2,"CR"+str(i)+"b",bp,bc2)
            import fitz
            t1="".join(p.get_text("text")for p in fitz.open(p1))
            t2="".join(p.get_text("text")for p in fitz.open(p2))
            fp1=extract_contacts_from_text(t1); fp2=extract_contacts_from_text(t2)
            ok=len(set(fp1.credit_codes)&set(fp2.credit_codes))>0
            TIMING.append(('credit_code',i,time.time()-t0))
            _save_result('credit_code',ok==expect,i,name,ok,expect,
                f"A={fp1.credit_codes} B={fp2.credit_codes}")
        except Exception as e:
            TIMING.append(('credit_code',i,time.time()-t0))
            _except('credit_code',i,name,e,"")


# ================================================================
# 报告
# ================================================================
def gen_report():
    rpath = os.path.join(TMPDIR,"test_report.txt")
    lines = ["="*80,"投标文件串标围标检测 — 全功能测试报告",
             f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             f"临时目录: {TMPDIR}","="*80]
    dims = {'content_similarity':'内容相似度（文本）','file_id':'文件码雷同','author':'文档作者雷同',
            'editor':'编辑经办人雷同','contact':'单位联系人雷同','company_name':'公司名称雷同','credit_code':'信用代码雷同'}
    gt=gp=0
    lines.append("\n【维度汇总】")
    lines.append(f"{'维度':<20} {'总用例':>6} {'通过':>6} {'失败':>6} {'通过率':>8}")
    lines.append("-"*50)
    for dk,dl in dims.items():
        t,p=TOTAL_CNT.get(dk,0),PASS_CNT.get(dk,0); f,r=t-p,f"{p/t*100:.0f}%"if t>0 else"N/A"
        lines.append(f"{dl:<20} {t:>6} {p:>6} {f:>6} {r:>8}"); gt+=t; gp+=p
    lines.append("-"*50)
    lines.append(f"{'总计':<20} {gt:>6} {gp:>6} {gt-gp:>6} {f'{gp/gt*100:.0f}%'if gt>0 else'N/A':>8}")

    lines.append("\n\n【详细测试结果】")
    prev=""
    for dim,idx,ok,name,act,exp,det in RESULTS:
        if dim!=prev: lines.append(f"\n--- {dim} ---"); prev=dim
        lines.append(f"  [{['FAIL','PASS'][ok]}] #{idx} {name}")
        lines.append(f"        预期={'匹配'if exp==True else'不匹配'if exp==False else exp} 实际={'匹配'if act==True else'不匹配'if act==False else act}")
        lines.append(f"        详情: {det[:200]}")
    if ERRORS:
        lines.append("\n\n【异常错误】")
        for d,i,n,e in ERRORS: lines.append(f"  {d}[{i}] {n}: {e[:200]}")
    lines.append(f"\n总耗时: {sum(t for _,_,t in TIMING):.1f}s")
    with open(rpath,'w',encoding='utf-8') as f: f.write('\n'.join(lines))
    print(f"\n测试完成! 总用例={gt} 通过={gp} 失败={gt-gp} 通过率={f'{gp/gt*100:.0f}%'if gt>0 else'N/A'}")
    print(f"报告: {rpath}")
    fails=[r for r in RESULTS if not r[2]]
    if fails:
        print("\n失败用例:")
        for d,i,_,n,_,_,det in fails: print(f"  [{d}#{i}] {n}: {det[:120]}")
    shutil.rmtree(TMPDIR,ignore_errors=True)
    print("临时文件已清理")


# ================================================================
# 入口
# ================================================================
def main():
    print("="*60)
    print("投标文件串标围标检测 — 全功能测试套件")
    print(f"临时目录: {TMPDIR}")
    print("="*60)
    for name,fn in [("文本相似度",test_text_sim),("文件码雷同",test_file_id),
                    ("文档作者",test_author),("编辑经办人",test_editor),
                    ("联系人雷同",test_contact),("公司名雷同",test_company),
                    ("信用代码",test_credit)]:
        print(f"\n▶ {name}")
        try: fn()
        except Exception as e: print(f"  [模块异常] {name}: {e}"); traceback.print_exc()
    gen_report()

if __name__=="__main__": main()
