"""Common customer questions answered by code, from data/*.json (owner ฟลุ๊ค).

The answer model (qwen3:4b) made up facts in the 2026-11-04 live test: an owner called
"คุณสมชาย", a shop called "ร้านอาหารไทยสุขสันต์", "ใช้ร้อนครึ่งได้" for คนละครึ่ง, a stock list
with a sold-out drink, and "เมนูอยู่ที่นี่ครับ" without a menu. Questions whose answer is in
our data, or that we know we cannot answer, never reach the model.

faq_answer(text) -> reply text, or None when the model should answer.
wants_order_summary(text) -> the customer asks to see the current order.
"""
from __future__ import annotations

import re
from typing import Optional

from .catalog import catalog, shop, stock

_ORDER_ID = re.compile(r"ORD-\d{8}-\d{4}", re.I)

_SUMMARY = re.compile(r"สรุป|ดู(ออเดอร์|ออร์เดอร์)|(ออเดอร์|ออร์เดอร์)(ของ|ผม|ฉัน|เรา)|ดูรายการ"
                      r"|สั่ง(อะไร|ไร)ไป")
_PEOPLE = re.compile(r"เจ้าของ|คนขาย|พนักงาน|แม่ค้า|พ่อค้า|กระจ๊อก|ลูกจ้าง|ลูกน้อง|แอดมิน|admin|คนทำ"
                     r"|เชฟ|คนส่ง|ไรเดอร์")
_PAYMENT = re.compile(r"จ่าย|ชำระ|โอน|พร้อมเพย์|promptpay|เงินสด|บัตร|คนละครึ่ง|ไทยช่วยไทย"
                      r"|เราเที่ยวด้วยกัน|qr|สแกน", re.I)
_HOW_TO = re.compile(r"ทำ(อะไร|ไร)ได้|ต้องแจ้ง|ต้องบอก|ต้องส่งอะไร|สั่ง(ยังไง|ไง|อย่างไร)|ใช้(ยังไง|งาน)"
                     r"|ไงต่อ|ยังไงต่อ|ทำยังไง|วิธีสั่ง|ช่วยอะไรได้")
_STOCK = re.compile(r"เหลือ(อะไร|ไร)|อะไรเหลือ|ไรเหลือ|(อะไร|ไร)หมด|หมดบ้าง|หมดยัง|ยังมี(อะไร|ไร)"
                    r"|มีของ|ของหมด")
_PRICE = re.compile(r"ราคา|เท่าไร|เท่าไหร่|กี่บาท")
_MENU = re.compile(r"เมนู|เมนุ|ขาย(อะไร|ไร)|มี(อะไร|ไร)(บ้าง|กิน)|มีอะไรให้สั่ง")
_FEE = re.compile(r"ค่าส่ง|ค่าจัดส่ง")
_HOURS = re.compile(r"เปิดกี่โมง|ปิดกี่โมง|เปิด(วัน|ถึง|ตอน)|ปิด(วัน|ตอน)|เวลาเปิด|เวลาทำการ")
_SHOP_NAME = re.compile(r"ร้าน(นี้)?ชื่อ|ชื่อร้าน|ชื่อ(อะไร|ไร)|คุณ(คือ)?ใคร|เป็นใคร")

UNKNOWN = "ขออภัยครับ ระบบไม่มีข้อมูลเรื่องนี้ เดี๋ยวพนักงานจะตอบกลับทาง LINE นี้ครับ"


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def wants_order_summary(text: str) -> bool:
    return bool(_SUMMARY.search(_squash(text)))


def mentions_old_order(text: str) -> bool:
    return bool(_ORDER_ID.search(text or ""))


def faq_answer(text: str) -> Optional[str]:
    """A fixed, data-backed answer for a known kind of question; None = not one of them."""
    t = _squash(text)
    if not t:
        return None
    s = shop()
    if _PEOPLE.search(t):
        return ("ขออภัยครับ ระบบไม่มีข้อมูลชื่อเจ้าของหรือพนักงานของร้าน "
                "ถ้ามีเรื่องติดต่อร้าน พนักงานจะตอบกลับทาง LINE นี้ครับ")
    if _PAYMENT.search(t):
        if s.get("payment"):
            return f"ร้านรับชำระเงินแบบ {s['payment']} ครับ"
        return ("ตอนนี้ระบบยังไม่มีข้อมูลช่องทางชำระเงินและโครงการส่วนลดครับ "
                "พนักงานจะแจ้งให้ทราบทาง LINE นี้ครับ")
    if _HOW_TO.search(t):
        return how_to_order()
    if _STOCK.search(t):
        return stock_text()
    if _PRICE.search(t):
        priced = [it for it in catalog().values()
                  if any(_squash(n) in t for n in (it.name, *it.aliases))]
        if priced:
            return " ".join(f"{it.name} {it.price} บาท" for it in priced) + " ครับ"
        return menu_reply()
    if _MENU.search(t):
        return menu_reply()
    if _FEE.search(t):
        return f"ค่าส่ง {s['delivery_fee']} บาทครับ ถ้ารับเองที่ร้านไม่มีค่าส่งครับ"
    if _HOURS.search(t):
        return f"ร้านเปิด {s['hours']} ครับ" if s.get("hours") else UNKNOWN
    if _SHOP_NAME.search(t):
        return f"ร้าน {s['name']} ครับ ผมเป็นผู้ช่วยรับออร์เดอร์ของร้านครับ"
    return None


def how_to_order() -> str:
    return ("สั่งอาหารได้ตามนี้ครับ\n"
            "1. พิมพ์ชื่อเมนูและจำนวน เช่น \"กะเพราหมูกรอบ 2\" หรือส่งรูปรายการอาหาร\n"
            "2. บอกว่ารับเองที่ร้าน หรือให้ไปส่ง\n"
            "3. ให้เบอร์โทร 10 หลัก (ถ้าให้ไปส่ง ขอที่อยู่พร้อมรหัสไปรษณีย์ด้วย)\n"
            "4. ตรวจสรุปออร์เดอร์ แล้วพิมพ์ \"ยืนยัน\" ครับ")


def menu_lines(mark_sold_out: bool = True) -> list[str]:
    """One line per dish for the customer, built from the catalog and live stock."""
    left = stock()
    rows = []
    for it in catalog().values():
        extra = ["/".join(v) for v in it.options.values()]
        extra += [f"+{a} {p} บาท" for a, p in it.addons.items()]
        sold_out = " (หมด)" if mark_sold_out and left.get(it.sku, 0) <= 0 else ""
        rows.append(f"- {it.name} {it.price} บาท{' (' + ', '.join(extra) + ')' if extra else ''}"
                    f"{sold_out}")
    return rows


def menu_reply() -> str:
    return "\n".join(["เมนูของร้านครับ", *menu_lines(),
                      "พิมพ์ชื่อเมนูและจำนวนที่ต้องการได้เลยครับ"])


def stock_text() -> str:
    left = stock()
    have = [f"{it.name} (เหลือ {left[it.sku]})" for it in catalog().values()
            if left.get(it.sku, 0) > 0]
    out = [it.name for it in catalog().values() if left.get(it.sku, 0) <= 0]
    if not have:
        return "ขออภัยครับ ตอนนี้ทุกเมนูหมดแล้วครับ"
    rows = ["ตอนนี้ยังสั่งได้: " + ", ".join(have)]
    if out:
        rows.append("หมดแล้ว: " + ", ".join(out))
    return "\n".join(rows) + " ครับ"


def shop_facts() -> str:
    """Facts for the answer prompt ($shop): only what is in data/shop.json."""
    s = shop()
    rows = [f"shop name: {s['name']}", f"delivery fee: {s['delivery_fee']} บาท (pickup: free)"]
    rows.append(f"payment: {s['payment']}" if s.get("payment") else "payment: unknown")
    rows.append(f"opening hours: {s['hours']}" if s.get("hours") else "opening hours: unknown")
    rows.append("owner and staff names: unknown")
    return "\n".join(rows)
