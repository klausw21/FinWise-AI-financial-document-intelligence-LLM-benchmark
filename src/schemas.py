"""Canonical extraction schemas (the single source of truth).

Field names mirror the dataset labels exactly so metrics can compare key-by-key.
All fields are Optional so that partial LLM output still validates (a missing
field is scored as wrong, not an exception). `strict_json_schema()` produces the
Claude structured-output schema (additionalProperties:false, no nullable unions).
"""
from __future__ import annotations

from typing import Literal, Optional, Type

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ---------- line-item / ledger rows ----------
class InvoiceItem(_Base):
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None


class ReceiptItem(_Base):
    name: Optional[str] = None
    qty: Optional[float] = None
    unit_price: Optional[str] = None    # dataset stores "$18.27" (string)
    line_total: Optional[str] = None


class BankTxnRow(_Base):
    date: Optional[str] = None
    description: Optional[str] = None
    debit: Optional[float] = None       # amount out (None if not a debit)
    credit: Optional[float] = None      # amount in  (None if not a credit)
    balance: Optional[float] = None     # running balance after this row


class CreditTxnRow(_Base):
    date: Optional[str] = None
    description: Optional[str] = None
    amount: Optional[float] = None      # positive number
    type: Optional[Literal["charge", "payment"]] = None


# ---------- document schemas ----------
class BankStatement(_Base):
    category: Optional[str] = "bank_statement"
    statement_id: Optional[str] = None
    account_holder: Optional[str] = None
    account_number_masked: Optional[str] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None
    transactions: Optional[int] = None            # label carries a COUNT
    transaction_rows: list[BankTxnRow] = Field(default_factory=list)  # from doc/gold


class Receipt(_Base):
    category: Optional[str] = "receipt"
    receipt_id: Optional[str] = None
    merchant_name: Optional[str] = None
    date: Optional[str] = None
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    discount: Optional[float] = None
    total: Optional[float] = None
    currency: Optional[str] = None
    items: list[ReceiptItem] = Field(default_factory=list)


class Invoice(_Base):
    invoice_id: Optional[str] = None
    invoice_number: Optional[str] = None
    company_name: Optional[str] = None
    company_address: Optional[str] = None
    customer_name: Optional[str] = None
    billing_address: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    currency: Optional[str] = None
    payment_terms: Optional[str] = None
    status: Optional[str] = None
    items: list[InvoiceItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    discount: Optional[float] = None
    tax_rate: Optional[float] = None
    tax: Optional[float] = None
    shipping: Optional[float] = None
    total: Optional[float] = None
    amount_paid: Optional[float] = None
    balance_due: Optional[float] = None
    vendor_email: Optional[str] = None
    vendor_phone: Optional[str] = None
    po_number: Optional[str] = None


class CreditCardStatement(_Base):
    category: Optional[str] = "credit_card_statement"
    statement_id: Optional[str] = None
    card_number_masked: Optional[str] = None
    cardholder: Optional[str] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    previous_balance: Optional[float] = None
    new_balance: Optional[float] = None
    minimum_payment: Optional[float] = None
    payment_due_date: Optional[str] = None
    credit_limit: Optional[float] = None
    transactions: list[CreditTxnRow] = Field(default_factory=list)


SCHEMAS: dict[str, Type[_Base]] = {
    "bank_statement": BankStatement,
    "receipt": Receipt,
    "invoice": Invoice,
    "credit_card_statement": CreditCardStatement,
}

# Which key in each schema holds the scorable list of rows/items, and the row model.
LIST_FIELD: dict[str, tuple[str, Type[_Base]]] = {
    "bank_statement": ("transaction_rows", BankTxnRow),
    "receipt": ("items", ReceiptItem),
    "invoice": ("items", InvoiceItem),
    "credit_card_statement": ("transactions", CreditTxnRow),
}


def schema_for(doc_type: str) -> Type[_Base]:
    return SCHEMAS[doc_type]


def strict_json_schema(doc_type: str) -> dict:
    """Claude structured-output schema that clears all three compilation limits.

    Claude's structured outputs cap: (a) union-typed params at 16, (b) optional
    params (not in `required`) at 24, and (c) grammar compile cost, which grows with
    the *subset* branching of optional fields (~2^N). Pydantic renders every
    `Optional[...]` field as a nullable union — Invoice alone has 26, so keeping them
    blows (a); but making everything optional to avoid (a) then blows (b)/(c)
    ("Schema is too complex"). The only config that clears all three is the original
    minus nullability:

      * drop the `null` branch of every field                  -> 0 unions  (clears a)
      * ROOT object: `required` = all properties               -> 0 top-level optionals,
                                                                   linear grammar (clears b, c)
      * nested ROW objects ($defs): leave fields OPTIONAL       -> the model can omit the
                                                                   inapplicable side of a
                                                                   debit/credit row so it
                                                                   matches gold's null; only
                                                                   4-5 fields, so 2^4 grammar
                                                                   and <=5 global optionals.
    """
    schema = schema_for(doc_type).model_json_schema()

    def _denull(node: dict) -> None:
        # anyOf:[T, {"type":"null"}]  ->  inline T (the single non-null branch)
        branches = [b for b in node.get("anyOf", []) if b.get("type") != "null"]
        if "anyOf" in node and len(branches) == 1:
            del node["anyOf"]
            for k, v in branches[0].items():
                node.setdefault(k, v)
        # type:["string","null"]  ->  "string"
        if isinstance(node.get("type"), list):
            non_null = [t for t in node["type"] if t != "null"]
            if len(non_null) == 1:
                node["type"] = non_null[0]
        node.pop("default", None)   # structured outputs don't want defaults/titles
        node.pop("title", None)

    def _harden(node: dict, root: bool = False) -> None:
        if not isinstance(node, dict):
            return
        _denull(node)
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            if root:                                    # root: all required -> linear grammar
                node["required"] = list(node["properties"].keys())
            else:                                       # nested rows: optional -> can omit debit/credit
                node.pop("required", None)
        # recurse into nested defs/properties/items (all non-root)
        for key in ("properties", "$defs"):
            for v in (node.get(key) or {}).values():
                _harden(v)
        if "items" in node and isinstance(node["items"], dict):
            _harden(node["items"])

    _harden(schema, root=True)
    return schema


if __name__ == "__main__":
    import json

    for t in SCHEMAS:
        m = schema_for(t)()
        print(f"{t:22s} fields={list(m.model_dump().keys())}")
    def _counts(node):
        """(unions, optionals) — Claude caps them at 16 and 24 respectively."""
        un = opt = 0
        if isinstance(node, dict):
            if "anyOf" in node or isinstance(node.get("type"), list):
                un += 1
            if node.get("type") == "object":
                req = set(node.get("required", []))
                opt += sum(1 for k in node.get("properties", {}) if k not in req)
            for v in list(node.get("properties", {}).values()) + list(node.get("$defs", {}).values()):
                u, o = _counts(v); un += u; opt += o
            if isinstance(node.get("items"), dict):
                u, o = _counts(node["items"]); un += u; opt += o
        return un, opt

    print("\nstrict JSON schema — limits per type (Claude caps: unions<=16, optionals<=24):")
    for t in SCHEMAS:
        s = strict_json_schema(t)
        un, opt = _counts(s)
        root_all_req = set(s.get("required", [])) == set(s["properties"].keys())
        print(f"  {t:22s} unions={un}  optionals={opt:2d}  root_all_required={root_all_req}")
