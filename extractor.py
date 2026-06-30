"""
模块 A：文档解析与特征提取引擎
"""
import os
import re
import hashlib
import logging
from io import BytesIO
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pdfplumber
import jieba
import numpy as np
from PIL import Image
import imagehash

from data_structures import BidFeature, MetadataFeature, QuoteSignature
from config import DetectionConfig

logger = logging.getLogger(__name__)


class DocumentFeatureExtractor:
    """文档特征提取器"""

    def __init__(self, config: DetectionConfig):
        self.config = config
        # 加载停用词
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self) -> set:
        """加载停用词表"""
        if self.config.STOPWORDS_PATH and os.path.exists(self.config.STOPWORDS_PATH):
            with open(self.config.STOPWORDS_PATH, 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f)
        # 默认停用词
        return {'的', '了', '和', '是', '在', '有', '与', '为', '等', '及'}

    def batch_extract(self, file_paths: List[str]) -> List[BidFeature]:
        """批量提取特征（并行）"""
        features = []

        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            future_to_path = {
                executor.submit(self.extract_single, path): path
                for path in file_paths
            }

            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    feature = future.result()
                    if feature:
                        features.append(feature)
                        logger.info(f"成功提取特征: {os.path.basename(path)}")
                except Exception as e:
                    logger.error(f"提取特征失败 {path}: {e}")

        return features

    def extract_single(self, file_path: str) -> Optional[BidFeature]:
        """提取单个文档特征"""
        try:
            # 生成文档ID
            doc_id = self._generate_doc_id(file_path)
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)

            # 打开PDF
            with pdfplumber.open(file_path) as pdf:
                # 提取文本
                text_content, paragraphs = self._extract_text(pdf)
                text_length = len(text_content)

                # 判断是否为扫描版
                is_scanned = text_length < 100

                # 提取元数据
                metadata = self._extract_metadata(pdf)

                # 提取图片哈希
                image_hashes = self._extract_image_hashes(pdf)

                # 限制文本长度
                if text_length > self.config.MAX_TEXT_LENGTH:
                    text_content = text_content[:self.config.MAX_TEXT_LENGTH]
                    text_length = self.config.MAX_TEXT_LENGTH

                # 计算SimHash
                text_simhash = self._compute_simhash(text_content) if not is_scanned else ""

                # 计算段落MinHash
                paragraph_hashes = self._compute_paragraph_minhashes(paragraphs) if not is_scanned else []

                # 提取报价
                quotes = self._extract_quotes(text_content)
                quote_signature = self._compute_quote_signature(quotes)

                return BidFeature(
                    doc_id=doc_id,
                    filename=filename,
                    file_size=file_size,
                    text_content=text_content,
                    text_length=text_length,
                    text_simhash=text_simhash,
                    paragraph_hashes=paragraph_hashes,
                    metadata=metadata,
                    quotes=quotes,
                    quote_signature=quote_signature,
                    image_hashes=image_hashes,
                    is_scanned=is_scanned
                )

        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {e}")
            return None

    def _generate_doc_id(self, file_path: str) -> str:
        """生成文档唯一ID"""
        return hashlib.md5(file_path.encode('utf-8')).hexdigest()[:16]

    def _extract_text(self, pdf) -> Tuple[str, List[str]]:
        """提取文本内容和段落"""
        all_text = []
        paragraphs = []

        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)

        full_text = "\n".join(all_text)

        # 按空行分割段落
        raw_paragraphs = re.split(r'\n\s*\n', full_text)
        paragraphs = [p.strip() for p in raw_paragraphs if len(p.strip()) > 50]

        return full_text, paragraphs

    def _extract_metadata(self, pdf) -> MetadataFeature:
        """提取元数据"""
        metadata_dict = pdf.metadata or {}

        author = metadata_dict.get('/Author', '').strip()
        creator = metadata_dict.get('/Creator', '').strip()
        producer = metadata_dict.get('/Producer', '').strip()

        # 提取时间
        created_time = self._parse_pdf_date(metadata_dict.get('/CreationDate', ''))
        modified_time = self._parse_pdf_date(metadata_dict.get('/ModDate', ''))

        # 生成软件指纹
        software_fingerprint = self._generate_software_fingerprint(creator, producer)

        # 时间分桶
        time_bucket = ""
        if created_time:
            try:
                dt = datetime.fromisoformat(created_time)
                time_bucket = dt.strftime(self.config.TIME_BUCKET_FORMAT)
            except:
                pass

        return MetadataFeature(
            author=author,
            creator=creator,
            producer=producer,
            created_time=created_time,
            modified_time=modified_time,
            software_fingerprint=software_fingerprint,
            time_bucket=time_bucket
        )

    def _parse_pdf_date(self, date_str: str) -> str:
        """解析PDF日期格式为ISO 8601"""
        if not date_str:
            return ""

        # PDF日期格式: D:20260629103000+08'00'
        match = re.match(r"D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", date_str)
        if match:
            year, month, day, hour, minute, second = match.groups()
            try:
                dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
                return dt.isoformat()
            except:
                pass

        return ""

    def _generate_software_fingerprint(self, creator: str, producer: str) -> str:
        """生成软件指纹"""
        # 去除版本号
        creator_clean = re.sub(r'\d+\.\d+[\d.]*', '', creator).strip()
        producer_clean = re.sub(r'\d+\.\d+[\d.]*', '', producer).strip()

        fingerprint = f"{creator_clean} + {producer_clean}".lower()
        return fingerprint

    def _extract_image_hashes(self, pdf) -> List[str]:
        """提取PDF中所有图片的感知哈希"""
        image_hashes = []

        for page_num, page in enumerate(pdf.pages):
            try:
                # 尝试提取图片
                if hasattr(page, 'images'):
                    images = page.images
                    if not images:
                        continue
                    
                    for img_info in images:
                        try:
                            # 获取图片数据
                            if isinstance(img_info, dict):
                                # pdfplumber不同版本的API差异
                                stream = img_info.get('stream') or img_info.get('object')
                            else:
                                stream = img_info
                            
                            if stream and hasattr(stream, 'read'):
                                img_data = stream.read()
                                # 转换为PIL Image
                                img = Image.open(BytesIO(img_data))
                                # 计算感知哈希（使用pHash）
                                phash = imagehash.phash(img)
                                image_hashes.append(str(phash))
                            elif stream and isinstance(stream, bytes):
                                # 直接是字节数据
                                img = Image.open(BytesIO(stream))
                                phash = imagehash.phash(img)
                                image_hashes.append(str(phash))
                        except Exception as e:
                            logger.debug(f"提取单个图片哈希失败: {e}")
                            continue
            except Exception as e:
                logger.debug(f"处理页面{page_num}图片失败: {e}")
                continue

        return image_hashes

    def _compute_simhash(self, text: str) -> str:
        """计算64位SimHash（返回十六进制字符串）"""
        if not text:
            return "0" * 16  # 返回16位十六进制字符串

        # 分词
        words = jieba.cut(text)
        word_list = [w for w in words if w not in self.stopwords and len(w) > 1]

        if not word_list:
            return "0" * 16

        # 词频统计
        word_freq = {}
        for word in word_list:
            word_freq[word] = word_freq.get(word, 0) + 1

        # 计算SimHash
        v = [0] * 64
        for word, freq in word_freq.items():
            # 使用MD5生成词的哈希值
            word_hash = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)

            for i in range(64):
                bit = (word_hash >> i) & 1
                if bit:
                    v[i] += freq
                else:
                    v[i] -= freq

        # 生成SimHash（64位整数）
        simhash_int = 0
        for i in range(64):
            if v[i] > 0:
                simhash_int |= (1 << i)

        # 返回十六进制字符串（便于存储和比较）
        return format(simhash_int, '016x')

    def _compute_paragraph_minhashes(self, paragraphs: List[str]) -> List[str]:
        """计算段落MinHash签名（简化版）"""
        paragraph_hashes = []
        num_hash_functions = 128  # MinHash签名长度

        # 预定义多个哈希函数（使用不同种子）
        def make_hash_func(seed):
            def hash_func(s):
                return hash(s + str(seed)) % (2**32)
            return hash_func

        hash_funcs = [make_hash_func(i) for i in range(num_hash_functions)]

        for para in paragraphs:
            # 分词
            words = list(jieba.cut(para))
            words = set([w for w in words if w not in self.stopwords and len(w) > 1])

            if not words:
                continue

            # 计算MinHash签名
            minhash_signature = []
            for hash_func in hash_funcs:
                min_val = min((hash_func(w) for w in words), default=float('inf'))
                minhash_signature.append(min_val)

            # 转换为字符串（用逗号分隔）
            minhash_str = ','.join(map(str, minhash_signature))
            paragraph_hashes.append(minhash_str)

        return paragraph_hashes

    def _extract_quotes(self, text: str) -> List[float]:
        """提取报价金额"""
        quotes = []

        # 正则匹配金额
        # 匹配模式：(¥|￥|人民币|RMB)?\s*[\d,]+\.?\d*\s*(万元|元|万)?
        patterns = [
            r'[¥￥]\s*([\d,]+\.?\d*)\s*万元',
            r'[¥￥]\s*([\d,]+\.?\d*)\s*元',
            r'人民币\s*([\d,]+\.?\d*)\s*万元',
            r'人民币\s*([\d,]+\.?\d*)\s*元',
            r'RMB\s*([\d,]+\.?\d*)\s*万元',
            r'RMB\s*([\d,]+\.?\d*)\s*元',
            r'([\d,]+\.?\d*)\s*万元',
            r'([\d,]+\.?\d*)\s*元'
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    amount_str = match.group(1).replace(',', '')
                    amount = float(amount_str)

                    # 判断单位
                    if '万' in match.group(0):
                        amount *= 10000

                    quotes.append(amount)
                except:
                    continue

        # 去重并排序
        quotes = sorted(list(set(quotes)))
        return quotes

    def _compute_quote_signature(self, quotes: List[float]) -> QuoteSignature:
        """计算报价统计特征"""
        if not quotes:
            return QuoteSignature()

        # 尾数分布
        tail_distribution = {}
        integer_count = 0

        for quote in quotes:
            # 提取小数部分
            decimal_part = quote - int(quote)
            if decimal_part < 0.01:
                integer_count += 1
                tail_key = "00"
            else:
                tail_key = str(int(decimal_part * 100)).zfill(2)

            tail_distribution[tail_key] = tail_distribution.get(tail_key, 0) + 1

        integer_ratio = integer_count / len(quotes)

        # 统计特征
        mean = float(np.mean(quotes))
        std = float(np.std(quotes))

        return QuoteSignature(
            count=len(quotes),
            values=quotes,
            tail_distribution=tail_distribution,
            integer_ratio=integer_ratio,
            mean=mean,
            std=std
        )