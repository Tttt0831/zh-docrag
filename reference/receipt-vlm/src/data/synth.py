"""
增强版票据合成器 — Enhanced Receipt Synthesizer

支持的模板类型:
  - vat_general:    增值税普通发票
  - vat_special:    增值税专用发票（含抵扣联）
  - receipt_thermal: 热敏小票/收据（模拟超市/餐饮小票）
  - receipt_english: 英文收据（模拟海外小票）

增强特性:
  - 多样化的字体、字号、粗细
  - 纸张纹理 + 不均匀光照 + 阴影
  - 印章 / 二维码
  - 透视变换、折叠痕迹
  - 商品明细合计算（含折扣/税率变化）
  - 随机背景（纯白/微黄/灰色热敏纸）
  - 高度溢出自动截断保护
"""

import os
import io
import random
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Literal
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# 0. 字体配置
# ═══════════════════════════════════════════════════════════════════════

CHINESE_FONTS = [
    "C:\\Windows\\Fonts\\msyh.ttc",       # 微软雅黑
    "C:\\Windows\\Fonts\\simsun.ttc",     # 宋体
    "C:\\Windows\\Fonts\\simkai.ttf",     # 楷体
    "C:\\Windows\\Fonts\\simfang.ttf",    # 仿宋
    "C:\\Windows\\Fonts\\simhei.ttf",     # 黑体
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]

MONO_FONTS = [
    "C:\\Windows\\Fonts\\consola.ttf",
    "C:\\Windows\\Fonts\\cour.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]

ENGLISH_FONTS = [
    "C:\\Windows\\Fonts\\arial.ttf",
    "C:\\Windows\\Fonts\\segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

_font_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}

# ═══════════════════════════════════════════════════════════════════════
# 0b. 难度配置
# ═══════════════════════════════════════════════════════════════════════
#
# 合成数据「太好学」的根因不是噪声不够，而是捷径太多：字体固定、版式固定、
# 商户名闭集、日期单一格式、没有干扰项。模型可以靠位置和先验猜，不必读图。
# 下面每个开关对应一类捷径，difficulty 档位控制打开哪些。

DIFFICULTY_PRESETS = {
    # 复刻修复前的行为，用于对照实验
    "easy": dict(
        font_random=False, layout_jitter=0.0, distractor_p=0.0,
        open_merchant=False, date_formats=False, amount_formats=False,
        label_variants=False, degrade="medium", hand_p=0.0, hand_mark_p=0.0,
    ),
    "medium": dict(
        font_random=True, layout_jitter=0.5, distractor_p=0.5,
        open_merchant=True, date_formats=True, amount_formats=True,
        label_variants=True, degrade="medium", hand_p=0.55, hand_mark_p=0.20,
    ),
    "hard": dict(
        font_random=True, layout_jitter=1.0, distractor_p=0.9,
        open_merchant=True, date_formats=True, amount_formats=True,
        label_variants=True, degrade="heavy", hand_p=0.85, hand_mark_p=0.40,
    ),
}

# 当前生效配置（由 set_difficulty 修改）
CFG: Dict[str, Any] = dict(DIFFICULTY_PRESETS["medium"])


def set_difficulty(level: str) -> Dict[str, Any]:
    """切换难度档位，返回生效配置。"""
    if level not in DIFFICULTY_PRESETS:
        raise ValueError(f"未知难度 {level}，可选 {list(DIFFICULTY_PRESETS)}")
    CFG.clear()
    CFG.update(DIFFICULTY_PRESETS[level])
    return dict(CFG)


def _jit(base: int, frac: float = 0.25) -> int:
    """按 layout_jitter 强度抖动一个尺寸/间距值。"""
    amp = CFG["layout_jitter"] * frac
    if amp <= 0:
        return base
    return max(1, int(round(base * random.uniform(1 - amp, 1 + amp))))


_FILLER_CN = [
    "备注: ", "经办人: 王磊", "复核: 李娜", "验旧号: {n}", "机器编号: {n}",
    "业务类型: 一般业务", "结算方式: 电汇", "合同号: HT{n}", "项目编号: XM{n}",
]


def _filler(draw, x: int, y: int, font, p: float = None) -> int:
    """
    以概率 p 插一行无关内容，把下面所有字段整体下推。

    这是破坏「位置先验」的主要手段：否则每个字段的 y 坐标在整个数据集里
    几乎是常数，模型学坐标就能定位，根本不用读标签。
    """
    p = CFG["layout_jitter"] * 0.55 if p is None else p
    if random.random() >= p:
        return y
    txt = random.choice(_FILLER_CN).replace("{n}", str(random.randint(10**5, 10**7)))
    draw.text((x, y), txt, fill=(120, 120, 120), font=font)
    return y + _jit(18, 0.3)


# ═══════════════════════════════════════════════════════════════════════
# 0d. 手写
# ═══════════════════════════════════════════════════════════════════════
#
# 真实票据上收款人/复核/开票人几乎都是手签的，备注常是手写，还经常有圈
# 划、打勾、"已付款"之类的批注。系统字体里没有手写体，用
# scripts/fetch_handwriting_fonts.py 取的 OFL 开源手写字体。

_HW_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "fonts" / "handwriting"
# 潦草体用于签名（签名本来就难认），清晰体用于备注和金额这类要读的内容
_HW_CURSIVE = ["LiuJianMaoCao-Regular.ttf", "ZhiMangXing-Regular.ttf"]
_HW_LEGIBLE = ["LongCang-Regular.ttf", "MaShanZheng-Regular.ttf"]
_HW_EN = ["Caveat-Regular.ttf", "Kalam-Regular.ttf", "IndieFlower-Regular.ttf"]

_hw_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}


def _hw_pool(style: str) -> List[str]:
    names = {"cursive": _HW_CURSIVE, "legible": _HW_LEGIBLE, "en": _HW_EN}[style]
    return [str(_HW_DIR / f) for f in names if (_HW_DIR / f).exists()]


def has_handwriting() -> bool:
    return bool(_hw_pool("cursive") or _hw_pool("legible"))


def get_hand_font(size: int, style: str = "cursive") -> Optional[ImageFont.FreeTypeFont]:
    pool = _hw_pool(style) or _hw_pool("legible") or _hw_pool("cursive")
    if not pool:
        return None
    fp = random.choice(pool)
    key = (fp, size)
    if key not in _hw_cache:
        try:
            _hw_cache[key] = ImageFont.truetype(fp, size)
        except Exception:
            return None
    return _hw_cache[key]


def _ink() -> Tuple[int, int, int]:
    """圆珠笔/签字笔墨色：蓝、蓝黑、黑，带轻微色偏。"""
    base = random.choice([(28, 42, 120), (18, 28, 72), (30, 30, 34), (16, 40, 96)])
    return tuple(max(0, min(255, c + random.randint(-12, 12))) for c in base)


def hand_text(draw, xy, text: str, size: int = 18, style: str = "cursive",
              fill=None, slope: float = None) -> int:
    """
    手写体绘制：逐字加基线抖动和轻微倾斜，避免整行对得太齐像印刷。

    返回绘制宽度（0 表示没有手写字体可用，调用方应回退到印刷体）。
    """
    font = get_hand_font(size, style)
    if font is None:
        return 0
    fill = fill or _ink()
    slope = random.uniform(-0.06, 0.06) if slope is None else slope
    x, y0 = xy
    x0 = x
    for i, ch in enumerate(text):
        dy = random.uniform(-size * 0.07, size * 0.07) + (x - x0) * slope
        draw.text((x, y0 + dy), ch, fill=fill, font=font)
        try:
            adv = draw.textlength(ch, font=font)
        except Exception:
            adv = size * 0.9
        x += adv + random.uniform(-size * 0.04, size * 0.09)
    return int(x - x0)


def _wobble(pts, amp: float):
    return [(px + random.uniform(-amp, amp), py + random.uniform(-amp, amp))
            for px, py in pts]


def hand_mark(draw, kind: str, box, fill=None) -> None:
    """手写批注：圈划 / 打勾 / 下划线 / 划掉。box=(x1,y1,x2,y2)。"""
    x1, y1, x2, y2 = box
    fill = fill or _ink()
    w = max(1, int(min(3, (y2 - y1) / 8)))
    if kind == "circle":
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        # 留足余量：圈是「围住」不是「划掉」，压到数字上会伤可读性
        rx = (x2 - x1) / 2 * 1.30 + 6
        ry = (y2 - y1) / 2 * 1.75 + 3
        pts = []
        start = random.uniform(0, 0.6)
        for k in range(33):
            a = start + k / 32 * 2 * math.pi * random.uniform(1.0, 1.12)
            pts.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
        draw.line(_wobble(pts, 1.6), fill=fill, width=w, joint="curve")
    elif kind == "check":
        h = y2 - y1
        pts = [(x1, y1 + h * 0.55), (x1 + h * 0.42, y2), (x2, y1 - h * 0.25)]
        draw.line(_wobble(pts, 1.2), fill=fill, width=w + 1)
    elif kind == "underline":
        yy = y2 + random.uniform(1, 4)
        pts = [(x1 + random.uniform(-3, 3), yy),
               ((x1 + x2) / 2, yy + random.uniform(-2, 2)),
               (x2 + random.uniform(-3, 5), yy + random.uniform(-2, 2))]
        draw.line(_wobble(pts, 1.0), fill=fill, width=w)
    elif kind == "strike":
        yy = (y1 + y2) / 2
        draw.line(_wobble([(x1 - 2, yy), (x2 + 2, yy + random.uniform(-2, 2))], 1.0),
                  fill=fill, width=w)


_HW_NAMES_CN = ["张丽", "王强", "李明", "赵芳", "刘洋", "陈静", "王芳", "李强",
                "张敏", "陈明", "赵雪", "李娜", "王磊", "孙宇", "周涛", "吴敏",
                "郑华", "冯军", "许静", "何峰"]
_HW_NOTES_CN = ["已付款", "现金结算", "当面点清", "已核对", "收讫", "转账已到",
                "发票已开", "客户自取"]
_HW_NAMES_EN = ["Bob", "Amy", "Carol", "Dave", "Erin", "Frank", "Grace"]
_HW_NOTES_EN = ["Paid", "Thank you!", "Verified", "Cash", "Checked"]


def _top_band(draw, img, m: int, y: int, width: int, font, locale: str = "cn") -> int:
    """
    在最顶上插入随机高度的区带（空白 / 联次标记 / 条码 / 标语）。

    位置先验的诊断显示：画在顶部区块的字段（发票号码、日期、商户名）标准差
    只有 0.008，因为它们上方没有任何可变内容。这个函数就是给它们「上方」
    制造随机高度。
    """
    if not CFG["layout_jitter"]:
        return y
    y += random.randint(0, int(18 * CFG["layout_jitter"]))

    r = random.random()
    if r < 0.28:                                  # 条码带
        bw = random.randint(90, 190)
        bx = random.choice([m, width - m - bw, (width - bw) // 2])
        bh = random.randint(14, 30)
        cx = bx
        while cx < bx + bw - 2:
            w_ = random.choice([1, 1, 2, 3])
            if random.random() < 0.55:
                draw.rectangle((cx, y, cx + w_, y + bh), fill=(40, 40, 40))
            cx += w_ + random.choice([1, 2])
        y += bh + random.randint(4, 12)
    elif r < 0.52:                                # 联次 / 标语
        txt = (random.choice(["存根联", "记账联", "发票联", "抵扣联", "第二联"])
               if locale == "cn" else
               random.choice(["** CUSTOMER COPY **", "** MERCHANT COPY **", "DUPLICATE"]))
        tw_ = _text_w(draw, txt, font)
        bx = random.choice([m, width - m - tw_, (width - tw_) // 2])
        draw.text((bx, y), txt, fill=(150, 120, 120), font=font)
        y += random.randint(16, 26)
    elif r < 0.68:                                # 纯空白
        y += random.randint(10, 40)
    return y


def _pick(options: List[str], default_idx: int = 0) -> str:
    """label_variants 开启时随机挑一个说法，否则用默认。"""
    return random.choice(options) if CFG["label_variants"] else options[default_idx]


# ═══════════════════════════════════════════════════════════════════════
# 0c. 系统字体发现（字体固定是最大的捷径之一）
# ═══════════════════════════════════════════════════════════════════════

_discovered: Dict[str, List[str]] = {}

# 每类字体必须能渲染的字符集。fc-list 只要字体含基本拉丁就会把它算进
# :lang=en，于是 Dingbats(D050000L)、PowerlineSymbols、MathJax、泰文/高棉文/
# 马拉雅拉姆文字体全被选进来——它们画出来是乱码或豆腐块。
# 「PIL 能打开」不等于「有我们要画的字形」，必须按实际字符集校验。
_REQ_LATIN = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz$.,:%#-/()"
_REQ_CN = (
    "发票代码号金额税率合计价购销售方名称纳人识别开具日期打印时间"
    "货物应劳务服规格数量单位小折扣实收找零支付宝微信银联现购物票"
    "件商品退换货电话购买方抵扣联存根记账第二备注经办复核验旧机器"
    "编业类型一般结算式汇同合项目大写整元角分年月"
)


# cmap 覆盖检查挡不住符号字体：Dingbats(D050000L)、StandardSymbolsPS 这类
# 字体恰恰是把 ASCII 码位映射到符号字形的，cmap 查得到、画出来是垃圾；
# 泰文/高棉文/马拉雅拉姆文字体含拉丁但字形不像票据印刷体。所以除了 cmap，
# 还要按字族白名单收口。
_LATIN_FAMILIES = (
    "DejaVuSans", "DejaVuSerif", "DejaVuSansMono",
    "LiberationSans", "LiberationSerif", "LiberationMono", "LiberationSansNarrow",
    "NimbusSans", "NimbusRoman", "NimbusMonoPS", "NimbusSansNarrow",
    "FreeSans", "FreeSerif", "FreeMono",
    "Ubuntu-", "UbuntuMono", "NotoSans", "NotoSerif", "NotoMono",
    "P052", "C059", "URWBookman", "URWGothic",
)
_MONO_FAMILIES = (
    "DejaVuSansMono", "LiberationMono", "NimbusMonoPS",
    "FreeMono", "UbuntuMono", "NotoSansMono", "NotoMono",
)
# 发票/小票不会用发丝字重，也极少通篇斜体
_WEIGHT_DENY = ("Thin", "ExtraLight", "Italic", "Oblique", "BdIta")
# Ubuntu 家族用后缀码而非全词：-Th 发丝，-RI/-BI/-LI/-MI 各种斜体
_SUFFIX_DENY = ("-Th.", "-RI.", "-BI.", "-LI.", "-MI.", "-CI.")


def _family_ok(fp: str, kind: str) -> bool:
    base = os.path.basename(fp)
    if any(d in base for d in _WEIGHT_DENY) or any(base.endswith(d.rstrip(".") + "." + e)
                                                   for d in _SUFFIX_DENY
                                                   for e in ("ttf", "otf", "ttc")):
        return False
    if kind == "cn":
        return True                      # 中文候选全是 Noto CJK / ukai / uming
    fams = _MONO_FAMILIES if kind == "mono" else _LATIN_FAMILIES
    return base.startswith(fams)


def _font_covers(fp: str, required: str) -> bool:
    """字体的 cmap 是否覆盖 required 里的全部字符。"""
    try:
        from fontTools.ttLib import TTCollection, TTFont
        fonts = (TTCollection(fp).fonts if fp.lower().endswith(".ttc")
                 else [TTFont(fp, fontNumber=0, lazy=True)])
        for f in fonts:
            try:
                cps = set()
                for table in f["cmap"].tables:
                    cps |= set(table.cmap.keys())
                if all(ord(c) in cps for c in required):
                    return True
            finally:
                try:
                    f.close()
                except Exception:
                    pass
    except Exception:
        return False
    return False


def _discover_fonts(kind: str) -> List[str]:
    """
    枚举系统上真正能用来画票据的字体。

    原实现 _find_font 返回「第一个可用字体」并缓存，整个数据集共用一种字形，
    是模型最容易利用的捷径之一。这里枚举全部候选、按字符覆盖率过滤，
    每张图随机选一套。
    """
    if kind in _discovered:
        return _discovered[kind]

    import subprocess
    query = {"cn": ":lang=zh", "en": ":lang=en", "mono": ":spacing=100"}[kind]
    paths: List[str] = []
    try:
        out = subprocess.run(["fc-list", query, "file"], capture_output=True,
                             text=True, timeout=30).stdout
        for line in out.splitlines():
            fp = line.strip().rstrip(":").strip()
            if fp.lower().endswith((".ttf", ".ttc", ".otf")):
                paths.append(fp)
    except Exception:
        pass

    fallback = {"cn": CHINESE_FONTS, "en": ENGLISH_FONTS, "mono": MONO_FONTS}[kind]
    paths = paths or [f for f in fallback if os.path.exists(f)]

    required = _REQ_LATIN + (_REQ_CN if kind == "cn" else "")
    usable = []
    for fp in sorted(set(paths)):
        try:
            ImageFont.truetype(fp, 16)
        except Exception:
            continue
        if _family_ok(fp, kind) and _font_covers(fp, required):
            usable.append(fp)

    # 同一字体常有多个路径（如 /usr/share/fonts 与 texlive 各一份），按文件名去重
    seen, dedup = set(), []
    for fp in usable:
        b = os.path.basename(fp)
        if b not in seen:
            seen.add(b)
            dedup.append(fp)

    _discovered[kind] = dedup or [f for f in fallback if os.path.exists(f)]
    return _discovered[kind]


# 当前这张图使用的字体族（由 draw_receipt 在每张图开始时抽定）
_CUR_FONT: Dict[str, Optional[str]] = {"cn": None, "en": None, "mono": None}


def new_font_family() -> None:
    """为下一张图抽一套字体。font_random 关闭时退回固定字体。"""
    if not CFG["font_random"]:
        _CUR_FONT.update(cn=None, en=None, mono=None)
        return
    for kind in ("cn", "en", "mono"):
        pool = _discover_fonts(kind)
        _CUR_FONT[kind] = random.choice(pool) if pool else None


def _text_width(draw: ImageDraw.Draw, text: str, font: ImageFont.ImageFont) -> int:
    """跨版本兼容的文本宽度计算。"""
    if hasattr(draw, "textlength"):
        return int(draw.textlength(text, font=font))
    # Pillow < 10 fallback
    bbox = draw.textbbox((0, 0), text, font=font) if hasattr(draw, "textbbox") else None
    if bbox:
        return bbox[2] - bbox[0]
    return len(text) * font.size // 2


def _font_line_height(font: ImageFont.ImageFont) -> int:
    """获取字体的真实行高（ascent + descent）。"""
    # Pillow >= 10 有 font.getmetrics()
    if hasattr(font, "getmetrics"):
        ascent, descent = font.getmetrics()
        return ascent + descent
    # 回退：1.4 倍字号
    return int(font.size * 1.4)


def _find_font(font_list: List[str], size: int = 20) -> ImageFont.ImageFont:
    """从列表中查找第一个可用字体。"""
    key = (",".join(font_list), size)
    if key in _font_cache:
        return _font_cache[key]

    for fp in font_list:
        if fp and os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, size)
                _font_cache[key] = font
                return font
            except Exception:
                continue

    # fallback: PIL default
    try:
        font = ImageFont.load_default()
    except Exception:
        font = ImageFont.ImageFont()
    _font_cache[key] = font
    return font


def _font_of(kind: str, fallback_list: List[str], size: int) -> ImageFont.ImageFont:
    """优先用本张图抽中的字体，失败回落到原有查找逻辑。"""
    fp = _CUR_FONT.get(kind)
    if fp:
        key = (fp, size)
        if key in _font_cache:
            return _font_cache[key]
        try:
            font = ImageFont.truetype(fp, size)
            _font_cache[key] = font
            return font
        except Exception:
            pass
    return _find_font(fallback_list, size)


def get_cn_font(size: int = 20) -> ImageFont.ImageFont:
    return _font_of("cn", CHINESE_FONTS, size)


def get_mono_font(size: int = 16) -> ImageFont.ImageFont:
    return _font_of("mono", MONO_FONTS, size)


def get_en_font(size: int = 16) -> ImageFont.ImageFont:
    return _font_of("en", ENGLISH_FONTS, size)


# ═══════════════════════════════════════════════════════════════════════
# 1. 数据生成
# ═══════════════════════════════════════════════════════════════════════

_COMPANY_POOL = [
    ("北京神州数码科技有限公司", "91110000MA00123456"),
    ("上海浦东发展科技有限公司", "91310115MA00567890"),
    ("深圳华为信息技术有限公司", "91440300MA00987654"),
    ("广州天河软件股份有限公司", "91440101MA00321456"),
    ("杭州阿里巴巴云计算有限公司", "91330100MA00789012"),
    ("成都腾讯科技股份有限公司", "91510100MA00111222"),
    ("武汉烽火通信科技有限公司", "91420100MA00444333"),
    ("西安大唐网络科技有限公司", "91610131MA00555666"),
    ("南京紫金山实验室有限公司", "91320100MA00666777"),
    ("苏州工业园区精密制造公司", "91320594MA00888999"),
    ("天津滨海新区物流有限公司", "91120116MA00123000"),
    ("重庆两江新区电子商务有限公司", "91500000MA00456000"),
    ("长沙高新开发区文化传媒有限公司", "91430100MA00789000"),
    ("济南高新技术产业开发区生物医药有限公司", "91370100MA00012345"),
    ("青岛海洋科技股份有限公司", "91370200MA00678000"),
    ("厦门自贸区进出口贸易有限公司", "91350200MA00999000"),
    ("合肥高新区人工智能研究院", "91340100MA00222000"),
    ("郑州经济技术开发区装备制造公司", "91410100MA00555000"),
    ("福州软件园信息技术服务有限公司", "91350100MA00888000"),
    ("大连高新技术产业园区新材料科技有限公司", "91210231MA00111000"),
]

_TAX_RATES = [
    (0.06, "6%"),
    (0.09, "9%"),
    (0.13, "13%"),
    (0.03, "3% (小规模)"),
]

_GOODS_POOL = [
    ("办公用品", ["A4复印纸", "签字笔", "文件夹", "订书机", "计算器", "便签纸", "档案盒"]),
    ("电子设备", ["笔记本电脑", "显示器", "键盘", "鼠标", "固态硬盘", "移动电源", "USB集线器"]),
    ("技术服务", ["软件开发", "系统集成", "技术咨询", "数据迁移", "安全保障", "运维服务"]),
    ("咨询服务", ["管理咨询", "财务顾问", "法律咨询", "市场调研", "战略规划"]),
    ("培训服务", ["Python培训", "项目管理培训", "领导力培训", "技术认证培训"]),
    ("物流运输", ["国内快递", "国际运输", "仓储服务", "同城配送"]),
    ("印刷服务", ["名片印刷", "宣传册印刷", "展架制作", "礼品包装"]),
    ("租赁服务", ["场地租赁", "设备租赁", "车辆租赁", "展具租赁"]),
]

_ENGLISH_COMPANIES = [
    ("Walmart Supercenter", "123 Main St, Springfield, IL 62701"),
    ("Target Corporation", "456 Oak Ave, Minneapolis, MN 55402"),
    ("Costco Wholesale", "789 Park Blvd, Seattle, WA 98101"),
    ("Best Buy Electronics", "321 Tech Dr, San Jose, CA 95113"),
    ("Starbucks Coffee", "100 Pike St, Seattle, WA 98101"),
    ("Whole Foods Market", "200 Green St, Austin, TX 78701"),
    ("Home Depot", "500 Builders Ln, Atlanta, GA 30339"),
    ("CVS Pharmacy", "800 Health Way, Boston, MA 02110"),
]

_ENGLISH_ITEMS = [
    ("Milk 2%", 1, 4.29), ("Bread Whole Wheat", 1, 3.49), ("Eggs Large 12ct", 1, 5.99),
    ("Banana Organic", 1, 1.99), ("Chicken Breast", 1, 8.99), ("Rice 5lb", 1, 6.49),
    ("Coffee Ground", 1, 9.99), ("Orange Juice", 1, 4.49), ("Yogurt Greek", 2, 1.49),
    ("Paper Towels", 1, 7.99), ("Dish Soap", 1, 3.29), ("Hand Sanitizer", 1, 4.99),
    ("Notebook", 2, 2.99), ("Pens Pack", 1, 4.49), ("USB Cable", 1, 12.99),
    ("Headphones", 1, 29.99), ("Charger", 1, 19.99), ("Screen Protector", 1, 8.99),
]


# ── 组合式商户名：把 20 选 1 的闭集变成开集 ──
# 闭集商户名让 merchant_name 退化成 20 分类，模型不需要 OCR 就能靠先验命中。
_CN_CITY = [
    "北京", "上海", "深圳", "广州", "杭州", "成都", "武汉", "西安", "南京", "苏州",
    "天津", "重庆", "长沙", "济南", "青岛", "厦门", "合肥", "郑州", "福州", "大连",
    "宁波", "无锡", "佛山", "东莞", "昆明", "沈阳", "哈尔滨", "石家庄", "太原", "南昌",
]
_CN_ZONE = ["", "", "", "高新区", "经济开发区", "自贸区", "工业园区", "新区"]
_CN_BRAND = [
    "鼎峰", "恒通", "嘉和", "博远", "天泽", "万宁", "朗程", "华宇", "睿智", "锦程",
    "聚力", "宏图", "正邦", "泰盛", "瑞丰", "远航", "明德", "信诚", "卓越", "同辉",
    "启元", "广联", "世纪", "金鹏", "立诚", "众诚", "安泰", "领先", "创元", "东方",
]
_CN_INDUSTRY = [
    "科技", "信息技术", "网络科技", "电子商务", "文化传媒", "生物医药", "装备制造",
    "新材料", "物流", "进出口贸易", "人工智能", "软件", "精密制造", "环保工程",
    "建筑工程", "医疗器械", "供应链管理", "广告", "餐饮管理", "商贸",
]
_CN_ORG = ["有限公司", "有限公司", "股份有限公司", "有限责任公司"]

_EN_BRAND = [
    "Brightpath", "Northgate", "Silverline", "Greenfield", "Ironwood", "Blueridge",
    "Summit", "Harborview", "Redstone", "Clearwater", "Fairmont", "Oakhill",
    "Pinecrest", "Stonebridge", "Westbrook", "Kingsway", "Maplewood", "Lakeshore",
]
_EN_KIND = [
    "Market", "Supply Co.", "Grocers", "Pharmacy", "Electronics", "Hardware",
    "Trading Co.", "Provisions", "General Store", "Outfitters", "Wholesale",
]
_EN_STREET = ["Main St", "Oak Ave", "Park Blvd", "Cedar Ln", "Elm St", "Pine Rd",
              "Market St", "River Rd", "Hill Dr", "Lake Ave"]
_EN_CITY = [("Springfield", "IL"), ("Madison", "WI"), ("Auburn", "AL"), ("Salem", "OR"),
            ("Dover", "DE"), ("Bristol", "CT"), ("Clinton", "IA"), ("Fairview", "NJ")]


def _compose_cn_company() -> str:
    return (random.choice(_CN_CITY) + random.choice(_CN_ZONE)
            + random.choice(_CN_BRAND) + random.choice(_CN_INDUSTRY)
            + random.choice(_CN_ORG))


def _compose_en_company() -> Tuple[str, str]:
    """返回 (店名, 地址)，地址也随机生成，避免店名→地址的固定映射。"""
    name = f"{random.choice(_EN_BRAND)} {random.choice(_EN_KIND)}"
    city, st = random.choice(_EN_CITY)
    addr = f"{random.randint(10, 9999)} {random.choice(_EN_STREET)}, {city}, {st} {random.randint(10000, 99999)}"
    return name, addr


# ── 日期与金额的「表面形式」：图上怎么写，标注怎么归一 ──
# 标注统一为 ISO(YYYY-MM-DD) 与 float，图上换着花样写。这样模型必须解析并
# 归一化，而不能把看到的字符串原样抄下来。

_CN_MONTH_DAY_STYLES = ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "cn_pad", "cn_nopad", "%Y%m%d"]
_EN_DATE_STYLES = ["%Y-%m-%d", "%b %d, %Y", "%d %b %Y", "%B %d, %Y"]


def render_date(iso: str, locale: str = "cn") -> str:
    """把 ISO 日期渲染成图上要写的样子。date_formats 关闭时原样返回。"""
    if not CFG["date_formats"]:
        return iso
    d = datetime.strptime(iso, "%Y-%m-%d")
    style = random.choice(_CN_MONTH_DAY_STYLES if locale == "cn" else _EN_DATE_STYLES)
    if style == "cn_pad":
        return f"{d.year}年{d.month:02d}月{d.day:02d}日"
    if style == "cn_nopad":
        return f"{d.year}年{d.month}月{d.day}日"
    return d.strftime(style)


def date_surface_forms(iso: str) -> List[str]:
    """该 ISO 日期所有可能的图上写法（校验器用）。"""
    d = datetime.strptime(iso, "%Y-%m-%d")
    forms = {iso}
    for style in set(_CN_MONTH_DAY_STYLES) | set(_EN_DATE_STYLES):
        if style == "cn_pad":
            forms.add(f"{d.year}年{d.month:02d}月{d.day:02d}日")
        elif style == "cn_nopad":
            forms.add(f"{d.year}年{d.month}月{d.day}日")
        else:
            forms.add(d.strftime(style))
    return sorted(forms)


def render_amount(v: float, locale: str = "cn") -> str:
    """金额的图上写法。数字部分始终连续，便于校验。"""
    if not CFG["amount_formats"]:
        return f"¥ {v:,.2f}" if locale == "cn" else f"${v:,.2f}"
    num = random.choice([f"{v:,.2f}", f"{v:.2f}"])
    if locale == "cn":
        return random.choice([f"¥ {num}", f"￥{num}", f"{num}元", f"RMB {num}", num])
    return random.choice([f"${num}", f"$ {num}", f"{num} USD", num])


def amount_surface_forms(v: float) -> List[str]:
    return [f"{v:,.2f}", f"{v:.2f}"]


def _random_company() -> Tuple[str, str]:
    """返回 (公司名, 纳税人识别号)。

    识别号保留前 10 位（行政区划 + MA），后 8 位每次随机重新生成。
    _COMPANY_POOL 只有 20 家公司，若沿用池内固定税号，模型可以靠公司名
    直接背出税号，tax_id 这个字段就不再需要真的去读图。
    """
    if CFG["open_merchant"]:
        name = _compose_cn_company()
        tid = random.choice(_COMPANY_POOL)[1]
    else:
        name, tid = random.choice(_COMPANY_POOL)
    return name, tid[:10] + "".join(random.choice("0123456789") for _ in range(8))


def _generate_phone() -> str:
    prefixes = ["010", "021", "0755", "028", "0571", "025", "027", "029"]
    return f"{random.choice(prefixes)}-{random.randint(10000000, 99999999)}"


def _generate_bank_info() -> Tuple[str, str]:
    banks = [
        ("中国工商银行", "北京分行"),
        ("中国建设银行", "上海分行"),
        ("中国农业银行", "深圳分行"),
        ("中国银行", "广州分行"),
        ("招商银行", "杭州分行"),
        ("交通银行", "成都分行"),
    ]
    name, branch = random.choice(banks)
    account = "".join([str(random.randint(0, 9)) for _ in range(16)])
    return f"{name}{branch}", account


def _random_date(start_year: int = 2021, end_year: int = 2026) -> str:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).strftime("%Y-%m-%d")


def _random_invoice_no() -> str:
    code = random.randint(110000, 450000)
    no = random.randint(10000000, 99999999)
    return f"{code}{no:08d}"


def _random_tax_id() -> str:
    return _random_company()[1]


def _random_amount(min_v: float = 50, max_v: float = 50000) -> float:
    """
    真实金额的分布：数量级近似对数均匀（小额远多于大额），且有很强的
    整数/整角偏好。原来用 uniform 采样，大额和分位小数被均匀采到，
    模型见到的数字分布和真实票据差得很远。
    """
    lo, hi = math.log10(max(1e-3, min_v)), math.log10(max_v)
    v = 10 ** random.uniform(lo, hi)
    r = random.random()
    if r < 0.22:                      # 整百/整十
        step = random.choice([100, 50, 10])
        v = max(step, round(v / step) * step)
    elif r < 0.50:                    # 整元
        v = max(1, round(v))
    elif r < 0.66:                    # .9 / .5 / .99 这类定价尾数
        v = max(1, int(v)) + random.choice([0.9, 0.5, 0.99, 0.8, 0.95])
    return round(min(max(v, min_v), max_v), 2)


def generate_invoice_data(template: str = "vat_general") -> Dict[str, Any]:
    """
    生成单条票据标注数据（仅填充模板不覆写的字段）。

    VAT 发票模板会在渲染时根据商品明细重新计算 total_amount/tax_amount，
    所以这里只为小票模板生成合理的金额。

    约定：**绘制函数才是标注的最终来源** —— draw_receipt() 会把真正画到图上的
    值写回 data，因此必须先 draw 再取 target_json，顺序反了标注就不可学。
    """
    merchant_name, tax_id = _random_company()
    data = {
        "merchant_name": merchant_name,
        "date": _random_date(2021, 2026),
        "total_amount": 0.0,   # 由模板填充
        "tax_amount": 0.0,     # 由模板填充
        "tax_id": tax_id,
        "invoice_no": _random_invoice_no(),
    }

    # 小票模板可能缺失部分字段
    if template in ("receipt_thermal", "receipt_english"):
        if random.random() < 0.6:
            data["tax_id"] = None
        if random.random() < 0.5:
            data["invoice_no"] = None

    return data


# ═══════════════════════════════════════════════════════════════════════
# 2. 辅助绘制函数
# ═══════════════════════════════════════════════════════════════════════

class _DrawContext:
    """绘制上下文：封装 canvas / draw / 边距，并提供防溢出检查。"""

    def __init__(self, draw: ImageDraw.Draw, img: Image.Image, margin: int = 30):
        self.draw = draw
        self.img = img
        self.m = margin
        self.w, self.h = img.size
        self._bottom_margin = margin

    @property
    def content_width(self) -> int:
        return self.w - 2 * self.m

    def check_overflow(self, y: int, needed: int = 20):
        """如果 y + needed 超出画布高度，截断图像。"""
        if y + needed > self.h - self._bottom_margin:
            # 裁剪到当前 y + needed
            new_h = min(y + needed + self._bottom_margin, self.h)
            # 不实际裁剪，只是标记（调用方应自行处理）
            return False  # 未溢出
        return True  # 安全


def _text_w(draw: ImageDraw.Draw, text: str, font: ImageFont.ImageFont) -> int:
    return _text_width(draw, text, font)


def _draw_center_text(draw: ImageDraw.Draw, text: str, y: int,
                      font: ImageFont.ImageFont, fill=(0, 0, 0), w: int = 0) -> int:
    """居中绘制文本。返回文本宽度。"""
    tw = _text_w(draw, text, font)
    x = (w - tw) // 2 if w > 0 else 0
    draw.text((x, y), text, fill=fill, font=font)
    return tw


def _draw_hline(draw: ImageDraw.Draw, x1: int, y: int, x2: int,
                fill=(0, 0, 0), width: int = 1):
    """绘制水平线。"""
    draw.line([(x1, y), (x2, y)], fill=fill, width=width)


def _draw_table_row(draw: ImageDraw.Draw, x: int, y: int, col_widths: List[int],
                    row_h: int, texts: List[str], font, header: bool = False) -> int:
    """绘制表格一行。使用字体真实行高进行垂直居中。返回下一行 y。"""
    bg = (230, 230, 230) if header else (255, 255, 255)
    line_h = _font_line_height(font)
    cx = x
    prev = getattr(draw, "layer", None)
    for i, (cw, txt) in enumerate(zip(col_widths, texts)):
        rect = (cx, y, cx + cw, y + row_h)
        draw.rectangle(rect, fill=bg, outline=(180, 180, 180))
        tw = _text_w(draw, txt, font)
        tx = cx + (cw - tw) // 2
        ty = y + (row_h - line_h) // 2
        # 表头文字是预印的，和框线一起走预印层；数据行才是套打上去的
        if header and prev is not None:
            draw.layer = "pre"
        draw.text((max(tx, cx + 2), max(ty, y + 1)), txt, fill=(0, 0, 0), font=font)
        if header and prev is not None:
            draw.layer = prev
        cx += cw
    return y + row_h


def _render_items_table(draw: ImageDraw.Draw, x: int, y: int,
                        w: int, font, total_rows: int = 0) -> Tuple[int, List[Dict]]:
    """
    在发票上渲染商品明细表。返回 (下一行 y, 商品列表)。

    total_rows: 表格固定总行数，不足部分补空行（真实发票的做法）。
    """
    row_h = _jit(26, 0.15)
    if total_rows <= 0:
        total_rows = random.randint(5, 8)
    col_widths = [int(w * 0.35), int(w * 0.15), int(w * 0.15), int(w * 0.15), int(w * 0.20)]
    headers = ["货物或应税劳务、服务名称", "规格", "数量", "单价", "金额"]

    y = _draw_table_row(draw, x, y, col_widths, row_h, headers, font, header=True)

    cat, items = random.choice(_GOODS_POOL)
    n_items = random.randint(1, min(5, len(items)))
    selected = random.sample(items, n_items)

    goods = []
    for item_name in selected:
        qty = random.randint(1, 20)
        unit_price = _random_amount(5, 5000)
        amount = round(qty * unit_price, 2)
        spec = random.choice(["", "箱", "台", "套", "次", "个"])
        row_texts = [item_name, spec, str(qty),
                     f"{unit_price:.2f}", f"{amount:.2f}"]
        y = _draw_table_row(draw, x, y, col_widths, row_h, row_texts, font)
        goods.append({"name": item_name, "qty": qty, "unit_price": unit_price, "amount": amount})

    # 补空行到固定行数：真实增值税发票的商品栏是固定高度的表格，没填满也画
    # 空行。只画填充行会让明细少的票整个下半部空出一大块，很不像真票。
    for _ in range(max(0, total_rows - n_items)):
        y = _draw_table_row(draw, x, y, col_widths, row_h, [""] * len(col_widths), font)

    return y, goods


# ═══════════════════════════════════════════════════════════════════════
# 3. 印章 / 二维码
# ═══════════════════════════════════════════════════════════════════════

class _LayerDraw:
    """
    把绘制调用分流到「预印层」和「打印层」，用来模拟套打错位。

    真实增值税发票是空白票据上先印好红框线、表格、标题，开票时再用针式
    打印机把内容套打上去，所以内容相对框线整体偏移 1~3mm 并带轻微旋转，
    金额经常压线或偏出格子。我们原来是一次性画完，框线和内容完美对齐，
    这是「一眼假」最大的来源，也让「字段在格子内固定位置」成了捷径。

    分流规则：线/框/椭圆/多边形 → 预印层；文字 → 打印层。
    表头和红色大标题本身是预印的，用 layer 属性显式覆盖。
    """

    _SHAPE = {"line", "rectangle", "ellipse", "polygon", "arc", "chord", "pieslice"}

    def __init__(self, pre: Image.Image, prt: Image.Image):
        self._pre_img, self._prt_img = pre, prt
        self._pre, self._prt = ImageDraw.Draw(pre), ImageDraw.Draw(prt)
        self.layer = "auto"

    def _pick(self, name):
        if self.layer == "pre":
            return self._pre
        if self.layer == "prt":
            return self._prt
        return self._pre if name in self._SHAPE else self._prt

    def __getattr__(self, name):
        return getattr(self._pick(name), name)


def _compose_overprint(base: Image.Image, pre: Image.Image,
                       prt: Image.Image) -> Image.Image:
    """合成预印层与打印层，打印层带随机偏移/旋转/缩放。"""
    base.alpha_composite(pre)
    if CFG["layout_jitter"] > 0:
        j = CFG["layout_jitter"]
        ang = random.uniform(-0.55, 0.55) * j
        if abs(ang) > 0.03:
            prt = prt.rotate(ang, resample=Image.BICUBIC, expand=False)
        sc = 1.0 + random.uniform(-0.004, 0.004) * j
        if abs(sc - 1.0) > 1e-4:
            w0, h0 = prt.size
            prt = prt.resize((max(1, int(w0 * sc)), max(1, int(h0 * sc))), Image.BICUBIC)
        dx = int(random.uniform(-9, 9) * j)
        dy = int(random.uniform(-7, 7) * j)
    else:
        dx = dy = 0
    base.alpha_composite(prt, (dx, dy)) if (dx >= 0 and dy >= 0) else \
        base.alpha_composite(prt.crop((-min(dx, 0), -min(dy, 0), prt.width, prt.height)),
                             (max(dx, 0), max(dy, 0)))
    return base


def security_pattern(img: Image.Image, tint=(196, 92, 92)) -> None:
    """
    在票面上叠加防伪底纹（原地修改 img）。

    真实增值税发票的票面不是纯色：有细密的波浪线/菱形网格和重复的「发票」
    水印。除了真实感，它还给 OCR 制造真实存在的背景干扰——我们此前是纯色
    底，背景过于干净。
    """
    w, h = img.size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    style = random.choice(["wave", "wave", "lattice", "wave_text"])
    a = random.randint(14, 30)
    col = tuple(tint) + (a,)

    if style in ("wave", "wave_text"):
        step = random.randint(9, 16)
        amp = random.uniform(2.0, 5.0)
        period = random.uniform(26, 60)
        phase0 = random.uniform(0, 6.28)
        for i, y0 in enumerate(range(-10, h + 10, step)):
            pts = []
            ph = phase0 + i * 0.7
            for x in range(0, w + 6, 6):
                pts.append((x, y0 + amp * math.sin(x / period + ph)))
            d.line(pts, fill=col, width=1)
    else:
        step = random.randint(14, 24)
        for x in range(-h, w + h, step):
            d.line([(x, 0), (x + h, h)], fill=col, width=1)
            d.line([(x, h), (x + h, 0)], fill=col, width=1)

    if style == "wave_text":
        f = get_cn_font(random.randint(15, 22))
        txt = random.choice(["发票", "增值税发票", "防伪"])
        tw = _text_w(d, txt, f) + random.randint(40, 90)
        for yy in range(10, h, random.randint(52, 84)):
            for xx in range(random.randint(-30, 0), w, tw):
                d.text((xx, yy), txt, fill=tuple(tint) + (max(8, a - 10),), font=f)

    img.alpha_composite(layer) if img.mode == "RGBA" else \
        img.paste(Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"), (0, 0))


def _star_points(cx: float, cy: float, r: float, rot: float = -math.pi / 2):
    pts = []
    for k in range(10):
        rr = r if k % 2 == 0 else r * 0.42
        a = rot + k * math.pi / 5
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def _arc_text(d, cx, cy, radius, text, font, fill, start_a, end_a, flip=False):
    """把文字沿圆弧排布（真章的环形文字）。"""
    if not text:
        return
    n = len(text)
    for i, ch in enumerate(text):
        t = (i + 0.5) / n
        a = start_a + (end_a - start_a) * t
        # 逐字渲染到小图再旋转，才能让字随弧线转向
        size = font.size
        tile = Image.new("RGBA", (size * 2, size * 2), (0, 0, 0, 0))
        td = ImageDraw.Draw(tile)
        tw = td.textlength(ch, font=font)
        td.text(((size * 2 - tw) / 2, size * 0.5), ch, font=font, fill=fill)
        deg = -math.degrees(a) - 90 + (180 if flip else 0)
        tile = tile.rotate(deg, resample=Image.BICUBIC, expand=False)
        px = cx + radius * math.cos(a) - size
        py = cy + radius * math.sin(a) - size
        d.im_paste(tile, (int(px), int(py)))


class _PasteDraw:
    """给 ImageDraw 附加一个 alpha 粘贴方法，方便弧形文字合成。"""
    def __init__(self, layer):
        self.layer = layer
        self.d = ImageDraw.Draw(layer)

    def __getattr__(self, k):
        return getattr(self.d, k)

    def im_paste(self, tile, pos):
        self.layer.alpha_composite(tile, pos)


def _draw_stamp(draw: ImageDraw.Draw, cx: int, cy: int, r: int,
                img: Image.Image = None, company: str = None):
    """
    绘制印章。

    原实现是一个完美的红色圆圈加水平文字，而且刻意避开正文。真章的特征是：
    环形文字沿弧排列、中间五角星、印油浓淡不均甚至断线、盖得歪、而且
    **盖在文字上**（半透明红压住黑字，字仍可读）。叠印是真实存在的遮挡难度。
    """
    if img is None:                      # 没有底图就退回简版，保证不崩
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(200, 40, 40), width=3)
        return

    SS = 2                                # 超采样，弧形文字才不毛糙
    size = int(r * 2.6) * SS
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    pd = _PasteDraw(layer)
    c = size / 2
    R = r * SS
    ink = random.choice([(198, 42, 42), (176, 34, 46), (206, 56, 48)])
    A = random.randint(150, 215)
    col = ink + (A,)

    # 外圈：半径带轻微抖动，并随机留 1~2 处断口（印油没压实）
    gaps = [(random.uniform(0, 6.28), random.uniform(0.12, 0.35))
            for _ in range(random.randint(0, 2))]
    prev = None
    for k in range(721):
        a = k / 720 * 2 * math.pi
        if any(abs(((a - g) + math.pi) % (2 * math.pi) - math.pi) < wdt for g, wdt in gaps):
            prev = None
            continue
        rr = R * (1 + 0.006 * math.sin(a * 3 + 1.1) + random.uniform(-0.004, 0.004))
        pt = (c + rr * math.cos(a), c + rr * math.sin(a))
        if prev:
            pd.line([prev, pt], fill=col, width=max(2, int(R * 0.055)))
        prev = pt

    # 中心五角星
    pd.polygon(_star_points(c, c, R * 0.30), fill=col)

    # 上弧：章名；下弧：单位名
    title = random.choice(["发票专用章", "财务专用章", "发票专用章"])
    f_big = get_cn_font(max(10, int(R * 0.26)))
    _arc_text(pd, c, c, R * 0.70, title, f_big, col, math.pi * 1.28, math.pi * 1.72)

    name = (company or _random_company()[0])
    name = name[:12]
    f_sm = get_cn_font(max(8, int(R * 0.16)))
    _arc_text(pd, c, c, R * 0.80, name, f_sm, col,
              math.pi * 0.78, math.pi * 0.22, flip=True)

    # 印油浓淡不均：用低频噪声调制 alpha，再随机旋转
    arr = np.array(layer).astype(np.float32)
    hh, ww = arr.shape[:2]
    n = np.random.rand(max(1, hh // 12), max(1, ww // 12)).astype(np.float32)
    n = np.array(Image.fromarray((n * 255).astype(np.uint8)).resize((ww, hh), Image.BICUBIC),
                 dtype=np.float32) / 255.0
    arr[:, :, 3] *= np.clip(0.55 + 0.75 * n, 0, 1.25)
    layer = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    layer = layer.rotate(random.uniform(-14, 14), resample=Image.BICUBIC, expand=False)
    layer = layer.resize((size // SS, size // SS), Image.LANCZOS)

    img.alpha_composite(layer, (int(cx - size / (2 * SS)), int(cy - size / (2 * SS))))


def _add_qr_code(draw: ImageDraw.Draw, x: int, y: int, size: int = 80,
                 img: Image.Image = None, payload: str = None):
    """
    绘制二维码。优先用 cv2 生成真实可解码的 QR，失败才退回模拟图案。

    载荷用真实增值税发票二维码的字段格式：
        01,发票类型,发票代码,发票号码,金额,开票日期,校验码
    原实现是随机方块——定位图案对了但数据区是噪声，模块密度和真 QR 的统计
    特性不一样，模型可能学到这个只在合成数据里成立的特征。
    """
    if payload is None:
        payload = "01,%02d,%s,%s,%.2f,%s,%s," % (
            random.choice([4, 10, 31]),
            f"{random.randint(110000, 450000)}{random.randint(10, 99)}",
            f"{random.randint(10000000, 99999999)}",
            round(random.uniform(50, 100000), 2),
            f"{random.randint(2021, 2026)}{random.randint(1,12):02d}{random.randint(1,28):02d}",
            "".join(random.choice("0123456789ABCDEF") for _ in range(6)),
        )

    if img is not None:
        try:
            import cv2
            enc = cv2.QRCodeEncoder_create()
            m = enc.encode(payload)
            if m is not None and m.size:
                arr = np.asarray(m)
                if arr.max() <= 1:
                    arr = arr * 255
                qr = Image.fromarray(arr.astype(np.uint8)).convert("L")
                # 留白边（quiet zone），再最近邻放大保持模块锐利
                qz = max(2, qr.size[0] // 12)
                canvas = Image.new("L", (qr.size[0] + 2 * qz, qr.size[1] + 2 * qz), 255)
                canvas.paste(qr, (qz, qz))
                canvas = canvas.resize((size, size), Image.NEAREST)
                img.paste(canvas.convert("RGB"), (x, y))
                return
        except Exception:
            pass

    # 回退：模拟图案
    rng_state = np.random.get_state()
    np.random.seed(abs(hash(str(x) + str(y))) % (2**31))
    draw.rectangle((x, y, x + size, y + size), fill=(255, 255, 255), outline=(0, 0, 0))
    for fx, fy in [(x + 4, y + 4), (x + size - 22, y + 4), (x + 4, y + size - 22)]:
        draw.rectangle((fx, fy, fx + 18, fy + 18), fill=(0, 0, 0))
        draw.rectangle((fx + 4, fy + 4, fx + 14, fy + 14), fill=(255, 255, 255))
        draw.rectangle((fx + 6, fy + 6, fx + 12, fy + 12), fill=(0, 0, 0))
    step = 4
    for i in range(size // step):
        for j in range(size // step):
            px, py = x + 1 + i * step, y + 1 + j * step
            in_finder = ((px < x + 24 and py < y + 24)
                         or (px > x + size - 26 and py < y + 24)
                         or (px < x + 24 and py > y + size - 26))
            if not in_finder and random.random() < 0.45:
                draw.rectangle((px, py, px + 2, py + 2), fill=(0, 0, 0))
    np.random.set_state(rng_state)


# ═══════════════════════════════════════════════════════════════════════
# 4. 模板: 增值税普通发票
# ═══════════════════════════════════════════════════════════════════════

def _draw_vat_general(data: Dict[str, Any], width: int = 0,
                      height: int = 0) -> Image.Image:
    """绘制增值税普通发票。"""
    # 画布尺寸随机：固定画布 + 固定版式 = 字段 y 坐标完全可预测
    if width <= 0:
        width = _jit(900, 0.10) if CFG["layout_jitter"] else 900
    if height <= 0:
        height = _jit(880, 0.10) if CFG["layout_jitter"] else 650
    img = Image.new("RGBA", (width, height),
                    color=(252, 250, 245, 255))
    security_pattern(img)
    # 预印层（框线/表格/标题）与打印层（全部可变内容）分开画，最后错位合成
    _pre = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    _prt = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = _LayerDraw(_pre, _prt)
    m = _jit(30)

    title_font = get_cn_font(_jit(26, 0.12))
    hdr_font = get_cn_font(_jit(16, 0.12))
    body_font = get_cn_font(_jit(13, 0.12))
    small_font = get_cn_font(_jit(11, 0.10))

    # ── 顶部 ──
    y = _top_band(draw, img, m, m + _jit(5, 1.5), width, small_font, "cn")
    code = f"{random.randint(110000, 450000)}{random.randint(10, 99)}"
    # 三个头部元素的左/中/右归属随机置换，破坏「发票号码总在右上角」这类先验
    head = [
        f"{_pick(['发票代码', '发票代码', '代码'])}: {code}",
        f"{_pick(['开票日期', '开票日期', '开 票 日 期'])}: {render_date(data['date'], 'cn')}",
        f"{_pick(['发票号码', '发票号码', '号码'])}: {data['invoice_no']}",
    ]
    if CFG["layout_jitter"]:
        random.shuffle(head)
    _draw_hline_x = [m, None, None]
    for slot, txt in enumerate(head):
        tw_h = _text_w(draw, txt, small_font)
        if slot == 0:
            hx = m
        elif slot == 1:
            hx = (width - tw_h) // 2
        else:
            hx = width - m - tw_h
        draw.text((hx, y), txt, fill=(80, 40, 40), font=small_font)
    y += _jit(20, 0.2)

    y = _filler(draw, m, y, small_font)

    # 干扰项：打印日期是另一个日期，标注要的是开票日期
    if random.random() < CFG["distractor_p"]:
        other = _random_date(2021, 2026)
        draw.text((m, y), f"{_pick(['打印日期', '打印时间', '录入日期'])}: "
                          f"{render_date(other, 'cn')}",
                  fill=(120, 90, 90), font=small_font)
        y += _jit(16, 0.2)

    title = "增值税普通发票        发票联"
    draw.layer = "pre"                       # 红色大标题是预印的
    _draw_center_text(draw, title, y, title_font, fill=(180, 30, 30), w=width)
    draw.layer = "auto"
    y += 40

    _draw_hline(draw, m, y, width - m, fill=(180, 30, 30), width=2)
    y += 12

    # ── 购买方信息 ──
    y = _filler(draw, m, y, small_font)
    draw.text((m, y), _pick(["购买方", "购买方", "购方信息"]), fill=(60, 60, 60), font=hdr_font)
    y += _jit(22, 0.25)
    buyer_name = _random_company()[0]
    buyer_tax_id = _random_tax_id()
    draw.text((m, y), f"名    称: {buyer_name}", fill=(0, 0, 0), font=body_font)
    y += 20
    draw.text((m, y), f"纳税人识别号: {buyer_tax_id}", fill=(0, 0, 0), font=body_font)
    draw.text((m + 350, y),
              f"地址、电话: {random.choice(_COMPANY_POOL)[0][:6]}区 {_generate_phone()}",
              fill=(0, 0, 0), font=small_font)
    y += 18
    bank, account = _generate_bank_info()
    draw.text((m, y), f"开户行及账号: {bank} {account}", fill=(0, 0, 0), font=small_font)
    y += _jit(26, 0.25)
    y = _filler(draw, m, y, small_font)

    _draw_hline(draw, m, y, width - m, fill=(180, 180, 180))
    y += 8

    # ── 商品明细表 ──
    y, goods = _render_items_table(draw, m, y, width - 2 * m, body_font,
                                   total_rows=random.randint(5, 8))
    y += 8

    # ── 金额合计 ──
    subtotal = sum(g["amount"] for g in goods)
    tax_rate = random.choice([0.03, 0.06, 0.09, 0.13])
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)
    data["total_amount"] = total
    data["tax_amount"] = tax

    info_x = width - m - 300
    info_w = 300
    draw.rectangle((info_x, y, info_x + info_w, y + 122),
                   outline=(180, 180, 180), fill=(255, 255, 252))
    row_h = 22
    info_rows = [
        (_pick(["金额合计（小写）", "金额合计", "合计金额"]), render_amount(subtotal)),
        (_pick(["税额", "税额", "税款金额"]), render_amount(tax)),
        ("价税合计（大写）", ""),
        ("（小写）", render_amount(total)),
    ]
    iy = y + 4
    total_box = None
    for ri, (label, val) in enumerate(info_rows):
        draw.text((info_x + 12, iy), label, fill=(60, 60, 60), font=small_font)
        if val:
            tw2 = _text_w(draw, val, small_font)
            vx = info_x + info_w - tw2 - 12
            draw.text((vx, iy), val, fill=(0, 0, 0), font=small_font)
            if ri == len(info_rows) - 1:
                # 记下合计金额的真实包围盒：render_amount 是随机的，
                # 再调一次得到的是另一种写法，宽度对不上。
                total_box = (vx, iy, vx + tw2, iy + row_h - 6)
        iy += row_h
    # 大写金额另起一行：原来画在 y+3*row_h，正好压在「（小写）」标签行上
    cn_amount = _num_to_chinese(total)
    draw.text((info_x + 12, y + 4 * row_h + 2), cn_amount, fill=(180, 30, 30), font=small_font)

    # 手写批注：把价税合计圈起来 / 划线，真实报销件上很常见
    if total_box and random.random() < CFG["hand_mark_p"]:
        hand_mark(draw, random.choice(["circle", "underline", "underline"]), total_box)

    y += 132

    # ── 销售方信息 ──
    _draw_hline(draw, m, y, width - m, fill=(180, 180, 180))
    y += _jit(10, 0.4)
    y = _filler(draw, m, y, small_font)
    # 销售方（开票方）就是标注里的商户，必须画标注值本身
    draw.text((m, y), f"{_pick(['销售方名称', '销方名称', '销 售 方', '销售方'])}: "
                      f"{data['merchant_name']}", fill=(0, 0, 0), font=body_font)
    y += _jit(20, 0.2)
    draw.text((m, y), f"{_pick(['纳税人识别号', '纳税人识别号', '统一社会信用代码', '税号'])}: "
                      f"{data['tax_id']}", fill=(0, 0, 0), font=body_font)
    y += 20
    draw.text((m, y),
              f"地址、电话: {random.choice(_COMPANY_POOL)[0][:4]}区 {_generate_phone()}",
              fill=(0, 0, 0), font=small_font)
    y += 18
    bank2, acct2 = _generate_bank_info()
    draw.text((m, y), f"开户行及账号: {bank2} {acct2}", fill=(0, 0, 0), font=small_font)
    y += 22

    _draw_hline(draw, m, y, width - m, fill=(180, 180, 180))
    y += 10
    # 收款人/复核/开票人：标签是印刷的，名字是手签的
    px = m
    for label in (_pick(["收款人", "收款人", "收 款 人"]),
                  _pick(["复核", "复核", "复 核"]),
                  _pick(["开票人", "开票人", "开 票 人"])):
        draw.text((px, y), f"{label}: ", fill=(60, 60, 60), font=small_font)
        lw = _text_w(draw, f"{label}: ", small_font)
        name = random.choice(_HW_NAMES_CN)
        if random.random() < CFG["hand_p"] and hand_text(
                draw, (px + lw, y - 4), name, size=_jit(17, 0.2), style="cursive"):
            pass
        else:
            draw.text((px + lw, y), name, fill=(60, 60, 60), font=small_font)
        px += _jit(200, 0.12)

    # ── 印章 + 二维码 ──
    # 填充行会把正文往下推，若仍按固定 height 偏移放置，印章/二维码会压住
    # 销售方名称和税号。检查器只看绘制调用、看不到遮挡，所以这里必须让它们
    # 主动避开正文底边。
    # 套打合成：打印层整体偏移/旋转后叠到预印层上
    img = _compose_overprint(img, _pre, _prt)
    post = ImageDraw.Draw(img)

    # 印章是开票后盖的，画在合成之后。它半透明，允许压住正文——真章就是这样，
    # 红色叠在黑字上字仍可读。但必须整枚落在票面内，露出半截等于没盖。
    stamp_r = _jit(80, 0.15)
    stamp_cx = min(width - stamp_r - 4, width - m - _jit(90, 0.15))
    stamp_cy = min(height - stamp_r - 4, height - _jit(105, 0.2))
    if stamp_cx > stamp_r and stamp_cy > stamp_r:
        _draw_stamp(post, stamp_cx, stamp_cy, stamp_r,
                    img=img, company=data.get("merchant_name"))

    # 二维码是不透明的，压住正文就是真丢信息，必须留在正文下方的空白区
    qr_size = _jit(75, 0.12)
    qr_y = max(y + 10, height - qr_size - _jit(30, 0.3))
    if qr_y + qr_size <= height - 4:
        _add_qr_code(post, m + _jit(10, 0.5), qr_y, qr_size, img=img)

    return img.convert("RGB")


# ═══════════════════════════════════════════════════════════════════════
# 5. 模板: 增值税专用发票
# ═══════════════════════════════════════════════════════════════════════

def _draw_vat_special(data: Dict[str, Any], width: int = 0,
                      height: int = 0) -> Image.Image:
    """绘制增值税专用发票（含抵扣联）。"""
    if width <= 0:
        width = _jit(950, 0.10) if CFG["layout_jitter"] else 950
    if height <= 0:
        height = _jit(830, 0.12) if CFG["layout_jitter"] else 680
    img = Image.new("RGBA", (width, height),
                    color=(252, 250, 245, 255))
    security_pattern(img)
    _pre = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    _prt = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = _LayerDraw(_pre, _prt)
    m = _jit(25)

    title_font = get_cn_font(_jit(28, 0.12))
    hdr_font = get_cn_font(_jit(15, 0.12))
    body_font = get_cn_font(_jit(12, 0.12))
    small_font = get_cn_font(_jit(10, 0.10))

    y = _top_band(draw, img, m, m + _jit(5, 1.5), width, small_font, "cn")
    no_txt = f"{_pick(['No', 'No', '发票号码'])}: {data['invoice_no']}"
    date_str = f"{_pick(['开票日期', '开票日期', '开具日期'])}: {render_date(data['date'], 'cn')}"
    # No / 日期 左右位置随机对调
    if CFG["layout_jitter"] and random.random() < 0.5:
        no_txt, date_str = date_str, no_txt
    draw.text((m, y), no_txt, fill=(80, 40, 40), font=small_font)
    tw_d = _text_w(draw, date_str, small_font)
    draw.text(((width - tw_d) // 2, y), date_str, fill=(80, 40, 40), font=small_font)
    tw = _text_w(draw, "抵扣联", hdr_font)
    draw.text((width - m - tw, y), "抵扣联", fill=(200, 40, 40), font=hdr_font)
    y += _jit(18, 0.3)
    y = _filler(draw, m, y, small_font)

    title = "××增值税专用发票"
    draw.layer = "pre"
    _draw_center_text(draw, title, y, title_font, fill=(180, 30, 30), w=width)
    draw.layer = "auto"
    y += 38

    _draw_hline(draw, m, y, width - m, fill=(180, 30, 30), width=3)
    y += _jit(10, 0.4)
    y = _filler(draw, m, y, small_font)

    buyer = _random_company()[0]
    draw.text((m, y), f"购货单位: {buyer}", fill=(0, 0, 0), font=body_font)
    draw.text((m + 400, y),
              f"纳税人识别号: {_random_tax_id()}", fill=(0, 0, 0), font=small_font)
    y += 20
    # 销货单位（开票方）就是标注里的商户
    draw.text((m, y), f"{_pick(['销货单位', '销方名称', '销售方'])}: {data['merchant_name']}",
              fill=(0, 0, 0), font=body_font)
    draw.text((m + 400, y),
              f"{_pick(['纳税人识别号', '税号', '统一社会信用代码'])}: {data['tax_id']}",
              fill=(0, 0, 0), font=small_font)
    y += 26

    _draw_hline(draw, m, y, width - m, fill=(180, 180, 180))
    y += 6
    y, goods = _render_items_table(draw, m, y, width - 2 * m, body_font,
                                   total_rows=random.randint(5, 8))
    y += 8

    subtotal = sum(g["amount"] for g in goods)
    tax = round(subtotal * 0.13, 2)
    total = round(subtotal + tax, 2)
    data["total_amount"] = total
    data["tax_amount"] = tax

    info_x = width - m - 320
    row_h = 20
    draw.rectangle((info_x, y, info_x + 320, y + 90),
                   outline=(180, 180, 180), fill=(255, 255, 252))
    iy = y + 4
    for label, val in [(_pick(["金额合计", "合计金额"]), render_amount(subtotal)),
                        (_pick(["税额", "税款金额"]), render_amount(tax)),
                        ("价税合计(大写)", _num_to_chinese(total)),
                        ("价税合计(小写)", render_amount(total))]:
        draw.text((info_x + 10, iy), label, fill=(60, 60, 60), font=small_font)
        tw2 = _text_w(draw, val, small_font)
        draw.text((info_x + 320 - tw2 - 10, iy), val, fill=(0, 0, 0), font=small_font)
        iy += row_h

    y += 100

    _draw_hline(draw, m, y, width - m, fill=(180, 180, 180))
    y += 8
    draw.text((m, y), "备注: ", fill=(100, 100, 100), font=small_font)
    # 备注栏手写
    if random.random() < CFG["hand_p"] * 0.6:
        hand_text(draw, (m + _text_w(draw, "备注: ", small_font) + 4, y - 4),
                  random.choice(_HW_NOTES_CN), size=_jit(16, 0.2), style="legible")
    y += _jit(24, 0.25)

    # 收款人/开票人手签（原实现两次 random.choice，量宽和实画的名字不是同一个）
    _n1, _n2 = random.choice(_HW_NAMES_CN), random.choice(_HW_NAMES_CN)
    draw.text((m, y), "收款人: ", fill=(60, 60, 60), font=small_font)
    _w1 = _text_w(draw, "收款人: ", small_font)
    if not (random.random() < CFG["hand_p"] and hand_text(
            draw, (m + _w1, y - 4), _n1, size=_jit(17, 0.2), style="cursive")):
        draw.text((m + _w1, y), _n1, fill=(60, 60, 60), font=small_font)

    _lbl2 = "开票人: "
    _w2 = _text_w(draw, _lbl2 + _n2, small_font)
    _x2 = width - m - _w2
    draw.text((_x2, y), _lbl2, fill=(60, 60, 60), font=small_font)
    if not (random.random() < CFG["hand_p"] and hand_text(
            draw, (_x2 + _text_w(draw, _lbl2, small_font), y - 4), _n2,
            size=_jit(17, 0.2), style="cursive")):
        draw.text((_x2 + _text_w(draw, _lbl2, small_font), y), _n2,
                  fill=(60, 60, 60), font=small_font)

    img = _compose_overprint(img, _pre, _prt)
    post = ImageDraw.Draw(img)

    stamp_r = _jit(90, 0.15)
    stamp_cx = min(width - stamp_r - 4, width - m - _jit(100, 0.15))
    stamp_cy = min(height - stamp_r - 4, height - _jit(95, 0.2))
    if stamp_cx > stamp_r and stamp_cy > stamp_r:
        _draw_stamp(post, stamp_cx, stamp_cy, stamp_r,
                    img=img, company=data.get("merchant_name"))

    return img.convert("RGB")


# ═══════════════════════════════════════════════════════════════════════
# 6. 模板: 热敏小票
# ═══════════════════════════════════════════════════════════════════════

def _draw_receipt_thermal(data: Dict[str, Any], width: int = 380,
                          height: int = 0) -> Image.Image:
    """绘制热敏小票（模拟超市/餐饮小票），高度自动适配内容。"""
    paper_color = random.choice([
        (250, 248, 242), (248, 246, 240), (252, 250, 245), (240, 238, 232)
    ])
    m = _jit(12)

    # 统一字体大小保证行对齐
    _base = _jit(14, 0.12)
    cn = get_cn_font(_base)
    cn_sm = get_cn_font(max(9, _base - 2))
    # 等宽数字字体与 cn_sm 大小一致确保同列对齐
    mono = get_mono_font(max(9, _base - 2))

    line_h_cn = _font_line_height(cn)
    line_h_sm = _font_line_height(cn_sm)

    # ── 先计算内容高度，再创建画布 ──
    cat, items = random.choice(_GOODS_POOL)
    n = random.randint(3, 8)
    selected = random.sample(items, min(n, len(items)))
    n_extra_lines = sum(1 for _ in [0] if random.random() < 0.4)  # discount line

    # 估算行数
    est_lines = (
        2 +  # 标题 + 分隔
        2 +  # 交易信息
        1 +  # 分隔
        1 +  # 表头
        1 +  # 表头下划线
        len(selected) +  # 商品行
        1 +  # 分隔
        3 + n_extra_lines +  # 合计行
        1 +  # 分隔
        1 +  # 支付行
        1 +  # 分隔
        2 +  # 底部
        4 +  # 干扰行（打印时间 / 会员号 / 找零）
        3    # 余量（版式抖动会吃高度）
    )
    est_height = est_lines * (line_h_sm + 6) + 30
    if height <= 0:
        height = max(360, min(est_height + 60, 1100))

    width = _jit(width, 0.12) if CFG["layout_jitter"] else width
    img = Image.new("RGB", (width, height), color=paper_color)
    draw = ImageDraw.Draw(img)

    y = _top_band(draw, img, m, _jit(16, 0.7), width, cn_sm, "cn")

    # ── 顶部 ──
    merchant = data["merchant_name"]
    _draw_center_text(draw, merchant, y, cn, w=width)
    y += line_h_cn + 6

    # 税号画在顶部而不是页脚：页脚前面有「内容画满就提前 return」的分支，
    # 画在那里会让一部分样本的 tax_id 标注落不到图上。
    if data["tax_id"]:
        _draw_center_text(draw, f"{_pick(['税号', '税号', '纳税人识别号'])}: {data['tax_id']}",
                          y, cn_sm, fill=(100, 100, 100), w=width)
        y += line_h_sm + 4

    _draw_center_text(draw, "—— 购物小票 ——", y, cn_sm, fill=(100, 100, 100), w=width)
    y += line_h_sm + 10

    _draw_hline(draw, m, y, width - m, fill=(100, 100, 100))
    y += 8

    # ── 交易信息 ──
    draw.text((m, y), f"{_pick(['日期', '日期', '交易日期'])}: {render_date(data['date'], 'cn')}",
              fill=(0, 0, 0), font=cn_sm)
    if data["invoice_no"]:
        draw.text((m + 140, y), f"{_pick(['小票号', '票号', '单号'])}: {data['invoice_no']}",
                  fill=(0, 0, 0), font=cn_sm)
    else:
        draw.text((m + 140, y), f"流水号: {random.randint(1000, 9999)}",
                  fill=(0, 0, 0), font=cn_sm)
    y += line_h_sm + 4

    # 干扰项：打印时间是另一个日期；会员号/流水号形似票号
    if random.random() < CFG["distractor_p"]:
        draw.text((m, y), f"打印时间: {render_date(_random_date(2021, 2026), 'cn')} "
                          f"{random.randint(9, 21):02d}:{random.randint(0, 59):02d}",
                  fill=(110, 110, 110), font=cn_sm)
        y += line_h_sm + 4
    if random.random() < CFG["distractor_p"]:
        draw.text((m, y), f"会员号: {random.randint(10**9, 10**10 - 1)}",
                  fill=(110, 110, 110), font=cn_sm)
        y += line_h_sm + 4
    draw.text((m, y), f"收银员: {random.choice(['001','002','003','005'])}",
              fill=(0, 0, 0), font=cn_sm)
    draw.text((m + 120, y), f"终端: POS{random.randint(10,99)}",
              fill=(0, 0, 0), font=cn_sm)
    y += line_h_sm + 4

    _draw_hline(draw, m, y, width - m, fill=(100, 100, 100))
    y += 6

    # ── 商品列表（统一用 cn_sm 保证中文正常、列宽一致） ──
    # 按内容宽度按比例分列：写死 m+170/260/320 是按 width=380 算的，
    # 宽度抖动后最右侧的「金额」列会被切出画布。
    _cw = width - 2 * m
    header_x = [m, m + int(_cw * 0.47), m + int(_cw * 0.70), m + int(_cw * 0.88)]
    draw.text((header_x[0], y), "商品", fill=(60, 60, 60), font=cn_sm)
    draw.text((header_x[1], y), "数量", fill=(60, 60, 60), font=cn_sm)
    draw.text((header_x[2], y), "单价", fill=(60, 60, 60), font=cn_sm)
    draw.text((width - m - _text_w(draw, "金额", cn_sm), y), "金额",
              fill=(60, 60, 60), font=cn_sm)
    y += line_h_sm + 2
    _draw_hline(draw, m, y, width - m, fill=(160, 160, 160))
    y += 6

    items_data = []
    for item_name in selected:
        if y + 20 > height - 40:
            break  # 防溢出
        qty = random.randint(1, 5)
        up = round(random.uniform(3, 500), 2)
        amt = round(qty * up, 2)
        # 中文商品名 cn_sm；数字也用 cn_sm（大小一致保证对齐）
        draw.text((header_x[0], y), item_name, fill=(0, 0, 0), font=cn_sm)
        draw.text((header_x[1] + 10, y), str(qty), fill=(0, 0, 0), font=cn_sm)
        draw.text((header_x[2] + 5, y), f"{up:.2f}", fill=(0, 0, 0), font=cn_sm)
        _amt_s = f"{amt:.2f}"
        draw.text((width - m - _text_w(draw, _amt_s, cn_sm), y), _amt_s,
                  fill=(0, 0, 0), font=cn_sm)
        y += line_h_sm + 4
        items_data.append({"name": item_name, "qty": qty, "unit_price": up, "amount": amt})

    y += 2
    _draw_hline(draw, m, y, width - m, fill=(100, 100, 100))
    y += 8

    # ── 合计 ──
    subtotal = sum(i["amount"] for i in items_data)
    discount = round(random.uniform(0, subtotal * 0.1), 2) if random.random() < 0.4 else 0
    after_discount = subtotal - discount
    tax_rate = random.choice([0, 0, 0, 0.06, 0.13])
    tax = round(after_discount * tax_rate, 2)
    total = round(after_discount + tax, 2)

    data["total_amount"] = total
    data["tax_amount"] = tax if tax > 0 else None

    items_count = sum(i["qty"] for i in items_data)
    draw.text((m, y + 2), f"件数: {items_count}", fill=(0, 0, 0), font=cn_sm)
    draw.text((m + 150, y + 2), f"小计: {render_amount(subtotal)}", fill=(0, 0, 0), font=cn_sm)
    y += line_h_sm + 4

    if discount > 0:
        draw.text((m + 150, y + 2), f"折扣: -{discount:.2f}", fill=(160, 40, 40), font=cn_sm)
        y += line_h_sm + 4
    if tax > 0:
        draw.text((m + 150, y + 2), f"税额: {tax:.2f}", fill=(0, 0, 0), font=cn_sm)
        y += line_h_sm + 4

    _draw_hline(draw, m + 120, y, width - m, fill=(60, 60, 60))
    y += 6
    total_str = f"{_pick(['合计', '合计', '应收', '总计'])}: {render_amount(total)}"
    tw_t = _text_w(draw, total_str, cn)
    draw.text((width - m - tw_t, y + 2), total_str, fill=(0, 0, 0), font=cn)
    y += line_h_cn + 10

    _draw_hline(draw, m, y, width - m, fill=(100, 100, 100))
    y += 8

    # ── 支付 ──
    payment_methods = ["微信支付", "支付宝", "银联卡", "现金", "云闪付"]
    pmt = random.choice(payment_methods)
    draw.text((m, y + 2), f"支付方式: {pmt}", fill=(0, 0, 0), font=cn_sm)
    # 干扰项：实收/找零是和 total 不同的数字，模型必须认准「合计」
    paid = round(total + random.choice([0, 0, round(random.uniform(1, 100), 2)]), 2)
    draw.text((m + 150, y + 2), f"实收: {render_amount(paid)}", fill=(0, 0, 0), font=cn_sm)
    y += line_h_sm + 4
    if paid > total and random.random() < CFG["distractor_p"]:
        draw.text((m + 150, y + 2), f"找零: {render_amount(round(paid - total, 2))}",
                  fill=(0, 0, 0), font=cn_sm)
        y += line_h_sm + 4
    y += 4

    if y > height - 40:
        return img  # 已满，不再画底部

    _draw_hline(draw, m, y, width - m, fill=(100, 100, 100))
    y += 8

    # ── 底部 ──
    # 手写批注：小票上常见「已付款」「收讫」之类
    if random.random() < CFG["hand_p"] * 0.5:
        note = random.choice(_HW_NOTES_CN)
        hw = hand_text(draw, (m + random.randint(0, 40), y - 2), note,
                       size=_jit(17, 0.2), style="legible")
        if hw:
            y += line_h_sm + 8

    _draw_center_text(draw, "请保留小票以办理退换货", y, cn_sm,
                      fill=(120, 120, 120), w=width)
    y += line_h_sm + 4
    _draw_center_text(draw, f"电话: {_generate_phone()}", y, cn_sm,
                      fill=(120, 120, 120), w=width)

    return img


# ═══════════════════════════════════════════════════════════════════════
# 7. 模板: 英文收据
# ═══════════════════════════════════════════════════════════════════════

def _draw_receipt_english(data: Dict[str, Any], width: int = 420,
                          height: int = 0) -> Image.Image:
    """绘制英文收据，高度自动适配内容。"""
    paper = random.choice([(252, 250, 245), (248, 246, 240), (255, 253, 248)])
    m = _jit(16)
    content_w = width - 2 * m  # 内容可用宽度
    right_x = width - m        # 右侧对齐基线

    _b = _jit(14, 0.12)
    en = get_en_font(_b)
    en_sm = get_en_font(max(9, _b - 3))
    mono = get_mono_font(max(10, _b - 1))
    bold_font = get_en_font(_jit(18, 0.12))

    line_h = _font_line_height(mono)
    line_h_sm = _font_line_height(en_sm)
    line_h_en = _font_line_height(en)
    line_h_bold = _font_line_height(bold_font)

    # 估算高度
    n_items = random.randint(3, 8)
    selected = random.sample(_ENGLISH_ITEMS, min(n_items, len(_ENGLISH_ITEMS)))
    est_lines = 20 + len(selected) + 6  # +3 干扰行 +3 抖动余量
    if height <= 0:
        height = max(350, min(est_lines * (line_h + 4) + 50, 1000))

    width = _jit(width, 0.12) if CFG["layout_jitter"] else width
    content_w = width - 2 * m
    right_x = width - m
    img = Image.new("RGB", (width, height), color=paper)
    draw = ImageDraw.Draw(img)

    def _truncate_item(text: str, max_px: int) -> str:
        """截断过长的商品名，确保不侵占价格列空间。"""
        if _text_w(draw, text, mono) <= max_px:
            return text
        while len(text) > 5 and _text_w(draw, text + "...", mono) > max_px:
            text = text[:-1]
        return text + "..."

    y = _top_band(draw, img, m, _jit(14, 0.8), width, en_sm, "en")

    # ── 商店信息 ──
    # 商户名以标注为准（此前这里直接随机挑一家，图上画的和标注无关）
    if CFG["open_merchant"]:
        company, address = _compose_en_company()
    else:
        addr_of = dict(_ENGLISH_COMPANIES)
        company = data.get("merchant_name")
        if company not in addr_of:
            company = random.choice(_ENGLISH_COMPANIES)[0]
        address = addr_of[company]
    data["merchant_name"] = company
    _draw_center_text(draw, company, y, bold_font, w=width)
    y += line_h_bold + 4
    _draw_center_text(draw, address, y, en_sm, fill=(80, 80, 80), w=width)
    y += line_h_sm + 2
    _draw_center_text(draw, "Tel: (555) 123-4567", y, en_sm, fill=(80, 80, 80), w=width)
    y += line_h_sm + 2
    # 英文收据用美式 EIN 税号，同样画在顶部（页脚有提前 return 分支）
    if data.get("tax_id"):
        data["tax_id"] = f"{random.randint(10, 99)}-{random.randint(1000000, 9999999)}"
        _draw_center_text(draw, f"Tax ID: {data['tax_id']}", y, en_sm,
                          fill=(80, 80, 80), w=width)
        y += line_h_sm + 2
    y += 4
    _draw_hline(draw, m, y, width - m, fill=(150, 150, 150))
    y += 8

    # ── 交易信息 ──
    dt_str = render_date(data["date"], "en") if data["date"] else "2024-06-15"
    draw.text((m, y), f"{_pick(['Date', 'Date', 'Sold'])}: {dt_str}", fill=(0, 0, 0), font=mono)
    txn_str = (f"{_pick(['Txn#', 'Txn#', 'Receipt#'])}: {data['invoice_no']}"
               if data.get("invoice_no") else f"Txn#: {random.randint(1000, 99999)}")
    tw = _text_w(draw, txn_str, mono)
    draw.text((right_x - tw, y), txn_str, fill=(0, 0, 0), font=mono)
    y += line_h + 2
    _cash_lbl = f"{_pick(['Cashier', 'Cashier', 'Served by'])}: "
    draw.text((m, y), _cash_lbl, fill=(60, 60, 60), font=mono)
    _cw2 = _text_w(draw, _cash_lbl, mono)
    _cn = random.choice(_HW_NAMES_EN)
    if not (random.random() < CFG["hand_p"] and hand_text(
            draw, (m + _cw2, y - 3), _cn, size=_jit(15, 0.2), style="en")):
        draw.text((m + _cw2, y), _cn, fill=(60, 60, 60), font=mono)
    reg_str = f"Reg: {random.randint(1, 20)}"
    tw = _text_w(draw, reg_str, mono)
    draw.text((right_x - tw, y), reg_str, fill=(60, 60, 60), font=mono)
    y += line_h + 2
    # 干扰项：Order# 形似 Receipt#，Printed 是另一个日期
    if random.random() < CFG["distractor_p"]:
        draw.text((m, y), f"Order#: {random.randint(10**7, 10**8 - 1)}",
                  fill=(120, 120, 120), font=mono)
        y += line_h + 2
    if random.random() < CFG["distractor_p"]:
        draw.text((m, y), f"Printed: {render_date(_random_date(2021, 2026), 'en')}",
                  fill=(120, 120, 120), font=mono)
        y += line_h + 2

    _draw_hline(draw, m, y, width - m, fill=(120, 120, 120))
    y += 6

    # ── 商品 ──
    # 预留价格列宽度（最宽价格 "$XX,XXX.XX" ≈ 80px）
    PRICE_COL_WIDTH = 80
    ITEM_MAX_WIDTH = content_w - PRICE_COL_WIDTH - 12  # 12px gap

    draw.text((m, y), "Item", fill=(100, 100, 100), font=en_sm)
    price_hdr = "Price"
    tw = _text_w(draw, price_hdr, en_sm)
    draw.text((right_x - tw, y), price_hdr, fill=(100, 100, 100), font=en_sm)
    y += line_h_sm + 2

    drawn_prices = []
    for name, qty, price in selected:
        if y + 20 > height - 40:
            break
        total_price = round(qty * price, 2)
        line = f"{name}" if qty == 1 else f"{name} x{qty}"
        line = _truncate_item(line, ITEM_MAX_WIDTH)
        price_str = f"${total_price:.2f}"
        draw.text((m, y), line, fill=(0, 0, 0), font=mono)
        tw = _text_w(draw, price_str, mono)
        draw.text((right_x - tw, y), price_str, fill=(0, 0, 0), font=mono)
        y += line_h + 2
        drawn_prices.append(total_price)

    y += 2
    _draw_hline(draw, m, y, width - m, fill=(120, 120, 120))
    y += 6

    # ── 合计（只统计实际画出来的行，溢出截断后金额才对得上） ──
    subtotal = round(sum(drawn_prices), 2)
    tax_pct = random.choice([0.06, 0.08, 0.0875, 0.0925])
    tax = round(subtotal * tax_pct, 2)
    total = round(subtotal + tax, 2)
    data["total_amount"] = total
    data["tax_amount"] = tax

    _rows = [(_pick(["Subtotal", "Subtotal", "Sub-total"]), subtotal),
             (f"Tax ({tax_pct*100:.2f}%)", tax),
             (_pick(["TOTAL", "TOTAL", "AMOUNT DUE"]), total)]
    for _ri, (label, val) in enumerate(_rows):
        val_s = render_amount(val, "en")
        is_total = (_ri == len(_rows) - 1)   # 最后一行才是合计，不能按标签文字判断
        val_font = bold_font if is_total else en
        val_line_h = _font_line_height(val_font)

        # 用实际渲染字体测量价格字符串宽度
        tw_val = _text_w(draw, val_s, val_font)
        tw_lbl = _text_w(draw, label, en)
        gap = 24  # label 和 value 之间的间距

        # value 右对齐，label 在 value 左边
        val_x = right_x - tw_val
        lbl_x = val_x - tw_lbl - gap

        draw.text((max(lbl_x, m), y), label, fill=(60, 60, 60), font=en)
        draw.text((val_x, y), val_s, fill=(0, 0, 0), font=val_font)
        y += max(line_h_en, val_line_h) + 4

    y += 2
    _draw_hline(draw, m, y, width - m, fill=(120, 120, 120))
    y += 6

    # ── 支付 ──
    methods = ["VISA ****1234", "MASTERCARD ****5678", "CASH", "DEBIT CARD", "AMEX ****9012"]
    pmt = random.choice(methods)
    draw.text((m, y), f"Payment: {pmt}", fill=(60, 60, 60), font=mono)
    y += line_h + 8

    if y > height - 30:
        return img

    # ── 底部 ──
    footer_lines = [
        "Thank you for shopping with us!",
        "Please retain receipt for returns within 30 days.",
        "Return Policy: exchanges accepted with receipt.",
    ]
    for fl in footer_lines:
        _draw_center_text(draw, fl, y, en_sm, fill=(140, 140, 140), w=width)
        y += line_h_sm + 2

    # 手写批注
    if random.random() < CFG["hand_p"] * 0.5:
        hand_text(draw, (m + random.randint(4, 50), y + 2),
                  random.choice(_HW_NOTES_EN), size=_jit(17, 0.2), style="en")

    return img


# ═══════════════════════════════════════════════════════════════════════
# 8. 中文大写金额
# ═══════════════════════════════════════════════════════════════════════

def _num_to_chinese(amount: float) -> str:
    """金额转中文大写。"""
    if amount >= 1e8:
        return "金额过大"

    digits = "零壹贰叁肆伍陆柒捌玖"
    units = ["", "拾", "佰", "仟"]
    big_units = ["", "万", "亿"]

    int_part = int(amount)
    dec_part = int(round((amount - int_part) * 100))

    if int_part == 0:
        result = "零元"
    else:
        result = ""
        int_str = str(int_part)
        n = len(int_str)
        zero_flag = False
        for i, ch in enumerate(int_str):
            d = int(ch)
            pos = n - i - 1
            unit_pos = pos % 4
            big_pos = pos // 4

            if d == 0:
                zero_flag = True
                if unit_pos == 0:
                    result += big_units[big_pos]
            else:
                if zero_flag:
                    result += "零"
                    zero_flag = False
                result += digits[d] + units[unit_pos]
                if unit_pos == 0:
                    result += big_units[big_pos]
        result += "元"

    jiao = dec_part // 10
    fen = dec_part % 10
    if jiao == 0 and fen == 0:
        result += "整"
    else:
        if jiao > 0:
            result += digits[jiao] + "角"
        if fen > 0:
            result += digits[fen] + "分"

    return result


# ═══════════════════════════════════════════════════════════════════════
# 9. 增强函数
# ═══════════════════════════════════════════════════════════════════════

def add_paper_texture(img: Image.Image, intensity: float = 0.03) -> Image.Image:
    """
    叠加纸张纹理。

    原实现对 RGB 三通道各自独立加高斯噪声（noise 的 shape 带通道维），出来的
    是彩色椒盐点，像传感器噪声而不是纸张颗粒，放大看就是一层彩色麻点。
    真实纸张颗粒是亮度上的、且有一定空间相关性，所以这里生成单通道噪声、
    做一次轻微低通，再等量加到三个通道上。
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    noise = np.random.normal(0, intensity * 255, (h, w)).astype(np.float32)
    # 3x3 均值低通：把逐像素白噪声变成有颗粒尺度的纹理
    k = np.ones((3, 3), np.float32) / 9.0
    pad = np.pad(noise, 1, mode="edge")
    sm = sum(pad[i:i + h, j:j + w] * k[i, j] for i in range(3) for j in range(3))
    noise = 0.45 * noise + 0.55 * sm * 1.6
    arr = np.clip(arr + noise[:, :, None], 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def add_uneven_lighting(img: Image.Image) -> Image.Image:
    """叠加不均匀光照（模拟拍照光照）。"""
    w, h = img.size
    arr = np.array(img, dtype=np.float32)

    y, x = np.ogrid[:h, :w]
    cx, cy = random.uniform(0.3, 0.7) * w, random.uniform(0.3, 0.7) * h
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_dist = np.sqrt(w**2 + h**2)
    light = 1.0 - random.uniform(0.05, 0.15) * (dist / max_dist)
    light = np.clip(light, 0.8, 1.0)

    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * light

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_fold_line(img: Image.Image) -> Image.Image:
    """添加折叠痕迹。"""
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    fold_x = random.randint(int(w * 0.3), int(w * 0.7))
    alpha = random.randint(20, 50)
    draw.line([(fold_x, 0), (fold_x, h)], fill=(180, 180, 180, alpha), width=2)

    arr = np.array(img, dtype=np.float32)
    arr[:, :fold_x, :] = np.clip(arr[:, :fold_x, :] * 1.02, 0, 255)
    arr[:, fold_x:, :] = np.clip(arr[:, fold_x:, :] * 0.98, 0, 255)
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def _perspective_coeffs(src, dst):
    """由四点对应解出 PIL PERSPECTIVE 所需的 8 个系数。"""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(dst, dtype=np.float64).reshape(8)
    return np.linalg.lstsq(A, b, rcond=None)[0].tolist()


def _desk_color(img: Image.Image) -> Tuple[int, int, int]:
    """票据周围的「桌面」底色，比纸略深。"""
    r, g, b = img.getpixel((0, 0))[:3]
    k = random.uniform(0.72, 0.92)
    return (int(r * k), int(g * k), int(b * k))


def _pad_for_geometry(img: Image.Image, frac: float) -> Image.Image:
    """
    几何变换前先扩边。

    不扩边的话，rotate(expand=False) 会切掉四角、透视会把边缘内容推出画布——
    顶部的发票号码/日期正好在边缘，一旦被裁掉，标注就不可学了。而
    check_synth_labels.py 只校验绘制阶段，抓不到增强阶段的丢失。
    """
    w, h = img.size
    pad = max(4, int(max(w, h) * frac))
    return ImageOps.expand(img, border=pad, fill=_desk_color(img))


def add_perspective(img: Image.Image, strength: float = 0.06,
                    pad: bool = True) -> Image.Image:
    """真四点透视（模拟手持拍摄的倾斜）。

    pad=False 用于管线内部——旋转和透视共用一次扩边，避免叠出嵌套边框。
    """
    if pad:
        img = _pad_for_geometry(img, strength * 1.2)
    w, h = img.size

    def j(v):
        return v * random.uniform(-strength, strength)

    dst = [(0, 0), (w, 0), (w, h), (0, h)]
    src = [(j(w), j(h)), (w + j(w), j(h)), (w + j(w), h + j(h)), (j(w), h + j(h))]
    try:
        coeffs = _perspective_coeffs(dst, src)
    except Exception:
        return img
    return img.transform((w, h), Image.PERSPECTIVE, coeffs,
                         resample=Image.BICUBIC, fillcolor=_desk_color(img))


def add_shadow(img: Image.Image) -> Image.Image:
    """从某一边渐暗的阴影，模拟手/机身遮光。"""
    w, h = img.size
    arr = np.array(img, dtype=np.float32)
    yy, xx = np.ogrid[:h, :w]
    ang = random.uniform(0, 2 * np.pi)
    proj = (np.cos(ang) * (xx / w) + np.sin(ang) * (yy / h))
    proj = (proj - proj.min()) / (np.ptp(proj) + 1e-6)   # numpy 2.x 移除了 ndarray.ptp
    depth = random.uniform(0.08, 0.24)   # 原来最深 0.38，压掉半边字
    mask = 1.0 - depth * np.clip((proj - random.uniform(0.2, 0.6)) * 2.2, 0, 1)
    arr *= mask[:, :, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_jpeg_artifacts(img: Image.Image, quality: int = None) -> Image.Image:
    """JPEG 压缩伪影——真实票据几乎都经过一次有损压缩。"""
    q = quality if quality is not None else random.randint(28, 72)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def add_resolution_loss(img: Image.Image, scale: float = None) -> Image.Image:
    """先降采样再放回原尺寸，模拟低分辨率扫描/远距离拍摄。"""
    w, h = img.size
    sc = scale if scale is not None else random.uniform(0.45, 0.8)
    small = img.resize((max(32, int(w * sc)), max(32, int(h * sc))), Image.BILINEAR)
    return small.resize((w, h), Image.BICUBIC)


def add_thermal_fading(img: Image.Image) -> Image.Image:
    """热敏纸褪色 + 打印头横向条纹。"""
    w, h = img.size
    arr = np.array(img, dtype=np.float32)
    # 整体发灰
    arr = arr * random.uniform(0.88, 0.97) + random.uniform(6, 22)
    # 横向条纹
    for _ in range(random.randint(1, 4)):
        y0 = random.randint(0, max(1, h - 6))
        hh = random.randint(1, 4)
        arr[y0:y0 + hh, :, :] *= random.uniform(1.04, 1.16)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_augmentation(img: Image.Image, level: str = "medium") -> Image.Image:
    """
    综合增强管道。

    退化按「拍摄质量档」分层，而不是每张图一律施加同等强度：
    真实票据照片里大多数是端正清晰的（人会主动对准拍），模糊、大角度倾斜、
    低分辨率是少数长尾。把每张图都弄糊弄斜既不真实，也会白白拉低可读性
    上限——模型学到的是「读不清时怎么猜」，不是「怎么读」。

    level 控制的是这三档的比例，不是强度：
      light  → 几乎全是 clean
      medium → clean 55% / normal 38% / poor 7%
      heavy  → clean 30% / normal 50% / poor 20%
    """
    mix = {
        "light":  (0.90, 0.99),
        "medium": (0.55, 0.93),
        "heavy":  (0.30, 0.80),
    }.get(level, (0.55, 0.93))
    r = random.random()
    tier = "clean" if r < mix[0] else ("normal" if r < mix[1] else "poor")

    # 每档的参数：角度(度)、透视强度、模糊半径、JPEG 质量区间
    P = {
        "clean":  dict(ang=0.8,  persp=0.0,   p_persp=0.0,  blur=0.0,
                       p_blur=0.0,  jpeg=(88, 97), p_jpeg=0.5,
                       p_light=0.25, p_shadow=0.0,  p_res=0.0,  p_fold=0.0),
        "normal": dict(ang=2.5,  persp=0.018, p_persp=0.35, blur=0.35,
                       p_blur=0.45, jpeg=(72, 92), p_jpeg=0.75,
                       p_light=0.55, p_shadow=0.30, p_res=0.10, p_fold=0.10),
        "poor":   dict(ang=5.0,  persp=0.045, p_persp=0.6,  blur=0.8,
                       p_blur=0.75, jpeg=(48, 75), p_jpeg=0.9,
                       p_light=0.7,  p_shadow=0.55, p_res=0.45, p_fold=0.25),
    }[tier]

    if random.random() < P["p_light"]:
        img = add_uneven_lighting(img)

    # ── 几何变换：旋转与透视共用一次扩边，结束后裁回 ──
    # 扩边是必须的（否则边缘字段会被切掉），但每个算子各扩一次会叠出嵌套
    # 灰框、票据在画面里越缩越小，所以这里统一处理。
    angle = random.uniform(-P["ang"], P["ang"])
    do_rot = abs(angle) > 0.15
    do_persp = P["p_persp"] > 0 and random.random() < P["p_persp"]
    p_str = P["persp"] * random.uniform(0.6, 1.0) if do_persp else 0.0

    if do_rot or do_persp:
        pad_frac = (abs(angle) / 90.0 if do_rot else 0.0) \
                   + (p_str * 1.2 if do_persp else 0.0) + 0.015
        img = _pad_for_geometry(img, pad_frac)

        # 超采样：PIL 的 draw.line/rectangle 不做抗锯齿，横线是硬边的，
        # 直接旋转会留下阶梯。先放大 2 倍做几何变换，再用 LANCZOS 缩回，
        # 相当于对阶梯做一次抗锯齿。
        tw, th = img.size
        img = img.resize((tw * 2, th * 2), Image.BICUBIC)
        if do_rot:
            # rotate 的默认 resample 是 NEAREST —— 这是「颗粒错位感」的主因
            img = img.rotate(angle, resample=Image.BICUBIC, expand=False,
                             fillcolor=_desk_color(img))
        if do_persp:
            img = add_perspective(img, strength=p_str, pad=False)
        img = img.resize((tw, th), Image.LANCZOS)

        # 裁回大部分留白，保留一点桌面背景（真实照片本来就有）
        w0, h0 = img.size
        cut = int(min(w0, h0) * pad_frac * random.uniform(0.55, 0.9))
        if cut > 0 and w0 - 2 * cut > 64 and h0 - 2 * cut > 64:
            img = img.crop((cut, cut, w0 - cut, h0 - cut))

    # 纹理放在几何变换之后：放在前面会被重采样拉成条纹，而且颗粒尺度会变
    img = add_paper_texture(img, intensity=random.uniform(0.004, 0.014))

    if P["p_blur"] > 0 and random.random() < P["p_blur"]:
        img = img.filter(ImageFilter.GaussianBlur(
            radius=random.uniform(0.15, P["blur"])))

    if P["p_shadow"] > 0 and random.random() < P["p_shadow"]:
        img = add_shadow(img)

    if P["p_res"] > 0 and random.random() < P["p_res"]:
        img = add_resolution_loss(img, scale=random.uniform(0.62, 0.85))

    if P["p_fold"] > 0 and random.random() < P["p_fold"]:
        img = add_fold_line(img)

    if tier != "clean" and random.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.12))
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.93, 1.08))

    # JPEG 放在最后：真实链路里压缩总是最后一环
    if random.random() < P["p_jpeg"]:
        img = add_jpeg_artifacts(img, quality=random.randint(*P["jpeg"]))

    return img


# ═══════════════════════════════════════════════════════════════════════
# 10. 主生成入口
# ═══════════════════════════════════════════════════════════════════════

TEMPLATE_MAP = {
    "vat_general": _draw_vat_general,
    "vat_special": _draw_vat_special,
    "receipt_thermal": _draw_receipt_thermal,
    "receipt_english": _draw_receipt_english,
}

TEMPLATE_WEIGHTS = {
    "vat_general": 0.35,
    "vat_special": 0.15,
    "receipt_thermal": 0.35,
    "receipt_english": 0.15,
}

DEFAULT_PROMPT = (
    "<image>请从这张票据中抽取以下字段并以 JSON 输出:\n"
    "merchant_name, date, total_amount, tax_amount, tax_id, invoice_no。\n"
    "缺失字段填 null,不要编造。"
)

ENGLISH_PROMPT = (
    "<image>Extract the following fields from this receipt as JSON:\n"
    "merchant_name, date, total_amount, tax_amount, tax_id, invoice_no.\n"
    "Use null for missing fields. Do not hallucinate."
)


def draw_receipt(data: Dict[str, Any],
                 template: str = "vat_general",
                 width: int = 0,
                 height: int = 0) -> Image.Image:
    """绘制票据图片。template='random' 随机选择模板。"""
    if template == "random":
        templates = list(TEMPLATE_WEIGHTS.keys())
        weights = list(TEMPLATE_WEIGHTS.values())
        template = random.choices(templates, weights=weights, k=1)[0]

    draw_fn = TEMPLATE_MAP.get(template)
    if draw_fn is None:
        raise ValueError(f"未知模板类型: {template}。可选: {list(TEMPLATE_MAP.keys())}")

    # 每张图换一套字体：字体固定是模型最容易利用的捷径之一
    new_font_family()

    kwargs = {}
    if width > 0:
        kwargs["width"] = width
    if height > 0:
        kwargs["height"] = height

    return draw_fn(data, **kwargs)


# ═══════════════════════════════════════════════════════════════════════
# 11. 批量生成
# ═══════════════════════════════════════════════════════════════════════

def generate_synthetic_dataset(
    output_dir: Path,
    num_samples: int = 2000,
    augmentation: str = "medium",
    seed: int = 42,
    difficulty: str = "medium",
) -> List[Dict[str, Any]]:
    """生成增强版合成票据数据集。

    difficulty 控制「捷径」的多少（见 DIFFICULTY_PRESETS）：
    easy 复刻旧行为用于对照，medium/hard 逐步关闭位置先验与闭集先验。
    """
    random.seed(seed)
    np.random.seed(seed)
    cfg = set_difficulty(difficulty)
    if augmentation == "auto":
        augmentation = cfg["degrade"]

    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    dataset = []
    templates = list(TEMPLATE_WEIGHTS.keys())
    weights = list(TEMPLATE_WEIGHTS.values())

    print(f"🎨 生成 {num_samples} 张增强合成票据...")
    print(f"   模板: {templates}")
    print(f"   权重: {weights}")
    print(f"   增强级别: {augmentation}")
    print(f"   难度: {difficulty}  {cfg}")

    for i in range(num_samples):
        tmpl = random.choices(templates, weights=weights, k=1)[0]
        data = generate_invoice_data(tmpl)
        if tmpl == "receipt_english" and not CFG["open_merchant"]:
            data["merchant_name"] = random.choice(_ENGLISH_COMPANIES)[0]

        # 绘制函数会把真正画上去的值写回 data，必须先画再取标注
        img = draw_receipt(data, template=tmpl)

        if tmpl == "receipt_thermal" and augmentation in ("medium", "heavy") \
                and random.random() < 0.45:
            img = add_thermal_fading(img)
        if augmentation != "none":
            img = apply_augmentation(img, level=augmentation)

        fname = f"synth_{i:06d}.png"
        fpath = image_dir / fname
        img.save(fpath)

        prompt = ENGLISH_PROMPT if tmpl == "receipt_english" else DEFAULT_PROMPT

        dataset.append({
            "image_path": str(fpath),
            "prompt": prompt,
            "target_json": data,
            "template": tmpl,
        })

        if (i + 1) % 200 == 0:
            print(f"  已生成 {i + 1}/{num_samples}")

    return dataset


# ═══════════════════════════════════════════════════════════════════════
# 12. CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    repo_root = Path(__file__).resolve().parent.parent.parent

    parser = argparse.ArgumentParser(description="增强版票据合成器")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="生成的样本数量；0 表示生成测试图")
    parser.add_argument("--output-dir", default=str(repo_root / "data" / "synthetic"),
                        help="输出目录")
    parser.add_argument("--augmentation",
                        choices=["none", "light", "medium", "heavy", "auto"],
                        default="auto", help="auto = 跟随 difficulty 档位")
    parser.add_argument("--difficulty", choices=list(DIFFICULTY_PRESETS),
                        default="medium")
    parser.add_argument("--template", choices=list(TEMPLATE_MAP.keys()) + ["random", "all"],
                        default="random")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    _cfg = set_difficulty(args.difficulty)
    if args.augmentation == "auto":
        args.augmentation = _cfg["degrade"]
    print(f"难度: {args.difficulty}  增强: {args.augmentation}\n")

    output_dir = Path(args.output_dir)

    if args.num_samples <= 0:
        print("生成各模板测试图...\n")
        output_dir.mkdir(parents=True, exist_ok=True)
        tmpls = list(TEMPLATE_MAP.keys()) if args.template == "all" else [args.template]

        for tmpl in tmpls:
            data = generate_invoice_data(tmpl)
            if tmpl == "receipt_english" and not CFG["open_merchant"]:
                data["merchant_name"] = random.choice(_ENGLISH_COMPANIES)[0]
            print(f"  模板: {tmpl}")
            print(f"  数据: {json.dumps(data, ensure_ascii=False, indent=2)}")

            img = draw_receipt(data, template=tmpl)
            if args.augmentation != "none":
                img = apply_augmentation(img, level=args.augmentation)

            out_path = output_dir / f"test_{tmpl}.png"
            img.save(out_path)
            print(f"  ✓ 保存到 {out_path}\n")
    else:
        dataset = generate_synthetic_dataset(
            output_dir=output_dir,
            num_samples=args.num_samples,
            augmentation=args.augmentation,
            seed=args.seed,
            difficulty=args.difficulty,
        )

        jsonl_path = output_dir / "train.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in dataset:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"\n✅ 生成 {len(dataset)} 条样本")
        print(f"   图片目录: {output_dir / 'images'}")
        print(f"   索引文件: {jsonl_path}")
