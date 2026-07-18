"""Business-facing field labels (EN/ZH) for the User view.

The schema uses database-style names (`invoice_id`, `company_name`); users should
see business language (`Document ID`, `Vendor`). Both languages are shipped so the
UI can re-label on a language toggle without re-analyzing.
"""
from __future__ import annotations

# field key -> (English label, 中文标签)
LABELS: dict[str, dict[str, tuple[str, str]]] = {
    "bank_statement": {
        "category": ("Type", "类型"),
        "statement_id": ("Statement ID", "对账单编号"),
        "account_holder": ("Account Holder", "账户持有人"),
        "account_number_masked": ("Account Number", "账号"),
        "period_start": ("Period Start", "起始日期"),
        "period_end": ("Period End", "结束日期"),
        "opening_balance": ("Opening Balance", "期初余额"),
        "closing_balance": ("Closing Balance", "期末余额"),
        "transactions": ("Transaction Count", "交易笔数"),
    },
    "receipt": {
        "category": ("Type", "类型"),
        "receipt_id": ("Receipt ID", "小票编号"),
        "merchant_name": ("Merchant", "商户"),
        "date": ("Date", "日期"),
        "subtotal": ("Subtotal", "小计"),
        "tax": ("Tax", "税费"),
        "discount": ("Discount", "折扣"),
        "total": ("Total Amount", "总金额"),
        "currency": ("Currency", "币种"),
    },
    "invoice": {
        "invoice_id": ("Document ID", "单据编号"),
        "invoice_number": ("Invoice Number", "发票号"),
        "company_name": ("Vendor", "供应商"),
        "company_address": ("Vendor Address", "供应商地址"),
        "customer_name": ("Bill To", "付款方"),
        "billing_address": ("Billing Address", "账单地址"),
        "invoice_date": ("Invoice Date", "开票日期"),
        "due_date": ("Due Date", "到期日"),
        "currency": ("Currency", "币种"),
        "payment_terms": ("Payment Terms", "付款条款"),
        "status": ("Status", "状态"),
        "subtotal": ("Subtotal", "小计"),
        "discount": ("Discount", "折扣"),
        "tax_rate": ("Tax Rate", "税率"),
        "tax": ("Tax", "税费"),
        "shipping": ("Shipping", "运费"),
        "total": ("Total Amount", "总金额"),
        "amount_paid": ("Amount Paid", "已付金额"),
        "balance_due": ("Balance Due", "应付余额"),
        "vendor_email": ("Vendor Email", "供应商邮箱"),
        "vendor_phone": ("Vendor Phone", "供应商电话"),
        "po_number": ("PO Number", "采购单号"),
    },
    "credit_card_statement": {
        "category": ("Type", "类型"),
        "statement_id": ("Statement ID", "对账单编号"),
        "card_number_masked": ("Card Number", "卡号"),
        "cardholder": ("Cardholder", "持卡人"),
        "period_start": ("Period Start", "起始日期"),
        "period_end": ("Period End", "结束日期"),
        "previous_balance": ("Previous Balance", "上期余额"),
        "new_balance": ("New Balance", "本期余额"),
        "minimum_payment": ("Minimum Payment", "最低还款额"),
        "payment_due_date": ("Payment Due Date", "还款到期日"),
        "credit_limit": ("Credit Limit", "信用额度"),
    },
}

# section title for the row/line-item table, per doc type
LIST_LABEL: dict[str, tuple[str, str]] = {
    "bank_statement": ("Transactions", "交易明细"),
    "credit_card_statement": ("Transactions", "交易明细"),
    "invoice": ("Line Items", "行项目"),
    "receipt": ("Line Items", "行项目"),
}


def field_label(doc_type: str, key: str) -> tuple[str, str]:
    """(en, zh) business label for a field; falls back to a prettified key."""
    lab = LABELS.get(doc_type, {}).get(key)
    if lab:
        return lab
    pretty = key.replace("_", " ").title()
    return (pretty, pretty)


def list_label(doc_type: str) -> tuple[str, str]:
    return LIST_LABEL.get(doc_type, ("Rows", "明细"))
