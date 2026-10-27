"""
ocr_extractor.py
Production-grade Multi-Image OCR & Structured Transaction Receipt Parser.
Supports PhonePe, Paytm, Google Pay (GPay), BHIM, Indian Bank IMPS/NEFT/RTGS receipts,
and International Bank Transfer Slips.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image
import pytesseract


KNOWN_BANKS = [
    "South Indian Bank",
    "Kerala Grameena Bank",
    "Karnataka Gramin Bank",
    "Kerala Gramin Bank",
    "The Federal Bank Ltd",
    "Federal Bank",
    "State Bank of India",
    "HDFC Bank",
    "ICICI Bank",
    "Axis Bank",
    "Canara Bank",
    "Punjab National Bank",
    "Bank of Baroda",
    "Union Bank of India",
    "Kotak Mahindra Bank",
    "IndusInd Bank",
    "Yes Bank",
    "IDFC First Bank",
    "IDBI Bank",
    "Indian Bank",
    "Indian Overseas Bank",
    "Central Bank of India",
    "Bank of India",
    "UCO Bank",
    "Bandhan Bank",
    "CSB Bank",
    "Dhanlaxmi Bank",
    "Karur Vysya Bank",
    "City Union Bank",
    "Tamilnad Mercantile Bank",
    "RBL Bank",
    "AU Small Finance Bank",
    "Equitas Small Finance Bank",
    "Ujjivan Small Finance Bank",
    "ESAF Small Finance Bank",
    "Jio Payments Bank",
    "Airtel Payments Bank",
    "India Post Payments Bank",
    "Paytm Payments Bank",
    "Fino Payments Bank",
    "Abu Dhabi Commercial Bank",
    "Emirates NBD",
    "First Abu Dhabi Bank",
    "Dubai Islamic Bank",
    "Mashreq Bank",
]


def preprocess_for_ocr(image_path: str) -> List[np.ndarray]:
    """
    Loads an image and prepares clean grayscale variants for Tesseract OCR.
    Preserves native resolution on standard screenshots so '₹' is not distorted.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        pil_img = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    if w < 600:
        scale = 800.0 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    variants = [gray, enhanced]
    if float(np.mean(gray)) < 95.0:
        inv = cv2.bitwise_not(gray)
        variants.append(inv)

    return variants


def clean_ifsc(code: str) -> str:
    """Normalizes 11-character Indian IFSC codes (first 4 letters, 5th char '0', remaining 6 chars with O->0)."""
    code = re.sub(r"[^A-Za-z0-9]", "", code).upper()
    if len(code) == 11:
        prefix = code[:4]
        branch = code[4:].replace("O", "0")
        return f"{prefix}{branch}"
    return code


def normalize_masked_account(raw_acc: str) -> str:
    """
    Normalizes masked or full account numbers where OCR misreads 'X' as 'K' or inserts spaces
    (e.g., 'XXXXXXKXXXKXX7243' or 'XXXXKXXXXXXXX7 243' -> 'XXXXXXXXXXXX7243').
    """
    s = raw_acc.strip()
    s = re.sub(r"(?<=[XK*0-9])\s+(?=\d{2,6}\b)", "", s, flags=re.IGNORECASE)
    if re.match(r"^[XK*x]{4,}[0-9]{3,8}$", s):
        prefix_len = len(re.match(r"^[XK*x]+", s).group(0))
        digits = s[prefix_len:]
        return ("X" * prefix_len) + digits
    return s


def clean_ocr_amount(raw_segment: str) -> float:
    """
    Sanitizes an amount string where OCR may misread '₹' as '2', 'z', 'Z', '7', '=', '?', 't', or 'R'.
    Also rejects dates, times (e.g. '01:37'), exchange rates ('1 AED = 1.0 AED'), and account numbers.
    """
    if not raw_segment:
        return 0.0

    s = str(raw_segment).strip()

    # Remove time/date fragments if present on the same line
    if re.search(r"\b\d{1,2}:\d{2}\b", s) and not re.search(r"[₹$,]\d", s):
        s = re.sub(
            r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\s*(?:on\s+\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4})?",
            " ",
            s,
            flags=re.IGNORECASE,
        )
    if re.search(r"=", s) and re.search(r"Exchange\s*Rate|\b1\s*[A-Z]{3}\s*=", s, flags=re.IGNORECASE):
        return 0.0

    had_explicit_marker = bool(
        re.match(r"^[\s:]*?(?:INR|Rs\.?|AED|USD|EUR|GBP|MRP|[₹$€£=~?tfFRzZ+\-*•]+)", s, flags=re.IGNORECASE)
    )

    # 1. Strip leading non-digit currency labels & OCR glyph noise
    s = re.sub(
        r"^(?:INR|Rs\.?|AED|USD|EUR|GBP|MRP|[\s:₹$€£=~?tfFRzZ+\-*•]+)+",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()

    # 2. If single leading '2' or '7' is separated by space from the number (e.g. '2 75,000.00' or '2 10,000')
    if re.match(r"^[27]\s+\d", s):
        s = re.sub(r"^[27]\s+", "", s).strip()
        had_explicit_marker = True

    # 3. Fix thousand-separator period/space OCR glitches
    s = re.sub(r"(\d)\.(\d{3})\.(\d{2})\b", r"\1,\2.\3", s)
    s = re.sub(r",\s+(?=\d{3}\b)", ",", s)

    # 4. Find the primary number token
    m = re.search(r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)", s)
    if not m:
        return 0.0

    num_str = m.group(1).rstrip(".")

    # 5. Strip fused leading '2' from Indian Rupee symbol '₹':
    # In Indian formatting, 6-digit numbers are written '2,10,000', NEVER '210,000' or '275,000.00'.
    # So '^2\d{2},\d{3}' (like '210,000' or '275,000.00') is ALWAYS '₹' fused onto a 5-digit amount ('10,000' or '75,000.00').
    if re.match(r"^2\d{2},\d{3}(?:\.[0-9]{1,2})?$", num_str):
        num_str = num_str[1:]
    elif re.match(r"^2\d{2},\d{2},\d{3}(?:\.[0-9]{1,2})?$", num_str):
        num_str = num_str[1:]
    elif not had_explicit_marker and re.match(r"^2(?:[1-9],\d{3}|\d{3,6})(?:\.[0-9]{1,2})?$", num_str):
        num_str = num_str[1:]

    try:
        val = float(num_str.replace(",", ""))
        if val >= 1000000000:
            return 0.0
        return val
    except ValueError:
        return 0.0


def parse_receipt_text(raw_text: str, filename: str = "") -> Dict[str, Any]:
    """
    Universal parser that extracts structured transaction fields from OCR text
    across PhonePe, Paytm, GPay, BHIM, Indian Banks, and International Bank Slips.
    """
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    joined = "\n".join(lines)

    # 1. Detect Payment App / Mode & Currency
    currency = "INR"
    if re.search(r"\bAED\b", joined):
        currency = "AED"
    elif re.search(r"\bUSD\b|\$", joined):
        currency = "USD"

    if re.search(r"PhonePe|\bT\d{18,25}\b", joined, re.IGNORECASE):
        payment_mode = "PhonePe (UPI)"
    elif re.search(r"Paytm|UPI\s*Ref\s*No", joined, re.IGNORECASE):
        payment_mode = "Paytm (UPI)"
    elif re.search(r"Google\s*Pay|GPay|Google\s*transaction\s*ID", joined, re.IGNORECASE):
        payment_mode = "Google Pay (UPI)"
    elif re.search(r"\bIMPS\b", joined):
        payment_mode = "IMPS"
    elif re.search(r"\bNEFT\b", joined):
        payment_mode = "NEFT"
    elif re.search(r"\bRTGS\b", joined):
        payment_mode = "RTGS"
    elif re.search(r"\bUPI\b", joined, re.IGNORECASE):
        payment_mode = "UPI"
    else:
        payment_mode = "Bank Transfer"

    # 2. Extract Status
    status = "Transaction Successful"

    # 3. Extract Transaction Amount (with cross-check & strict exclusion of dates/times/IDs)
    candidate_amounts: List[float] = []

    # Priority A: Explicit Amount Labels ('Transaction Amount', 'Transfer Amount', 'Total Amount', 'Amount Paid')
    label_patterns = [
        r"(?:Transaction\s*Amount|Transfer\s*Amount|Total\s*Amount|Amount\s*Paid|Paid\s*Amount|Debited\s*Amount)[\s\n:]*([^\n]+(?:\n[₹Rs.INRAEDUSDzZ27=?\s0-9,.]+)?)",
        r"\bAmount[\s\n:]+([₹Rs.INRAEDUSDzZ27=?\s0-9,]+(?:\.[0-9]{1,2})?)",
    ]
    for pat in label_patterns:
        for m in re.finditer(pat, joined, re.IGNORECASE):
            seg = m.group(1).strip()
            for subline in seg.splitlines():
                subline = subline.strip()
                if not subline or re.search(r"Exchange\s*Rate|Charges|Fee", subline, re.I):
                    continue
                val = clean_ocr_amount(subline)
                if val > 0:
                    candidate_amounts.append(val)
                    break

    # Priority B: PhonePe / Paytm / GPay 'Paid to <Name> <Amount>' or 'Debited from <Acc> <Amount>' or standalone currency line
    for i, line in enumerate(lines):
        if re.search(
            r"(?:Transaction\s*ID|Receipt\s*No|UTR|RRN|IFSC|Exchange\s*Rate|Charges|Fee|\b\d{1,2}:\d{2}\s*(?:AM|PM)\b|\bon\s+\d{1,2}\s+[A-Za-z]{3})",
            line,
            re.IGNORECASE,
        ):
            continue

        if re.match(r"^(?:Paid\s*to|Sent\s*to|Debited\s*from|To)\b", line, re.IGNORECASE):
            for offset in (0, 1, 2):
                if i + offset < len(lines):
                    target_line = lines[i + offset]
                    if re.search(r"\b\d{1,2}:\d{2}\b|UTR|Transaction\s*ID|Bank", target_line, re.I):
                        continue
                    trail_m = re.search(
                        r"(?:[₹$]|Rs\.?|INR|AED|\s)([27]?\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d{3,7}\.\d{2})\s*(?:AED|INR|Rs\.?)?\s*$",
                        target_line,
                        re.IGNORECASE,
                    )
                    if trail_m:
                        v = clean_ocr_amount(trail_m.group(0))
                        if v > 0:
                            candidate_amounts.append(v)

        explicit_curr = re.search(
            r"(?:₹|Rs\.?|INR)\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)|([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*(?:AED|INR|USD)\b",
            line,
            re.IGNORECASE,
        )
        if explicit_curr:
            v = clean_ocr_amount(explicit_curr.group(0))
            if v > 0:
                candidate_amounts.append(v)

    amount_val = 0.0
    if candidate_amounts:
        amount_set = set(int(round(a)) for a in candidate_amounts if a > 0)
        filtered = []
        for a in candidate_amounts:
            int_s = str(int(round(a)))
            if len(int_s) >= 4 and int_s[0] in ("2", "7") and int(int_s[1:]) in amount_set:
                continue
            filtered.append(a)
        amount_val = filtered[0] if filtered else candidate_amounts[0]

    # 4. Charges / Fees
    charges_val = 0.0
    chg_m = re.search(
        r"(?:IMPS\s*Charges|Bank\s*Charges|Platform\s*Fee|Convenience\s*Fee|Charges|Fee)[\s\n:₹Rs.=~zZ-]*([0-9]+(?:\.[0-9]{1,2})?)",
        joined,
        re.IGNORECASE,
    )
    if chg_m:
        try:
            charges_val = float(chg_m.group(1))
        except ValueError:
            charges_val = 0.0

    # 5. Transaction ID / Receipt No.
    txn_id = ""
    tid_patterns = [
        r"(?:PhonePe\s*Transaction\s*ID|Google\s*transaction\s*ID|Transaction\s*ID|Txn\s*ID|Trans\s*ID|Receipt\s*No\.?|Order\s*ID|Reference\s*ID)[\s\n:#-]*([A-Za-z0-9\-]{6,32})",
        r"\b(T\d{18,25})\b",
    ]
    for pat in tid_patterns:
        m = re.search(pat, joined, re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            if cand.upper() not in {"SUCCESS", "SUCCESSFUL", "COMPLETED", "TRANSACTION", "DETAILS"}:
                txn_id = cand
                break

    # 6. UTR / RRN / UPI Ref No
    rrn = ""
    rrn_patterns = [
        r"(?:Transaction\s*RRN|RRN\s*(?:No\.?)?|UTR\s*(?:No\.?|Number)?|UPI\s*Ref\.?\s*(?:No\.?|ID)?|UPI\s*transaction\s*ID|IMPS\s*Ref\.?\s*(?:No\.?)?)[\s\n:#-]*([0-9A-Za-z]{10,20})",
        r"\bUTR[\s:#-]*([0-9]{10,16})\b",
    ]
    for pat in rrn_patterns:
        m = re.search(pat, joined, re.IGNORECASE)
        if m:
            rrn = m.group(1).strip()
            break

    if not rrn:
        for line in lines:
            if not re.search(r"A/C|Account|Number|Mobile|Phone", line, re.IGNORECASE):
                m12 = re.search(r"\b(\d{12})\b", line)
                if m12 and m12.group(1) != txn_id:
                    rrn = m12.group(1)
                    break

    if not txn_id and rrn:
        txn_id = rrn
    if not rrn and txn_id:
        rrn = txn_id

    # 7. Date & Time
    date_time = "N/A"
    dt_patterns = [
        r"(\d{1,2}:\d{2}\s*(?:AM|PM)\s+on\s+\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})",
        r"(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\s*(?:,|at|\s)\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)",
        r"([A-Za-z]{3,9}\s+\d{1,2}\s*,?\s*\d{4}\s*(?:,|at|\s)\s*\d{1,2}:\d{2}\s*(?:AM|PM)?)",
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*,?\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)",
        r"(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})",
    ]
    for pat in dt_patterns:
        m = re.search(pat, joined, re.IGNORECASE)
        if m:
            raw_dt = m.group(1).strip()
            if not re.search(r"\b(?:AM|PM)\s+on\s+\d", raw_dt, re.IGNORECASE):
                raw_dt = re.sub(r"(\b(?:AM|PM)\b).*$", r"\1", raw_dt, flags=re.IGNORECASE)
            date_time = raw_dt.strip()
            break

    # 8. Beneficiary (Receiver) Details — Name, Account Number, Bank Name, IFSC
    receiver_name = "N/A"
    receiver_account = "N/A"
    receiver_ifsc = "N/A"
    receiver_bank = "N/A"

    # 8a. Check explicit beneficiary labels ('Banking Name :', 'Account Name', 'Account Number', 'Bank Name')
    bname_m = re.search(
        r"(?:Banking\s*Name|Account\s*Name|Beneficiary\s*Name)[\s:.\-]+([A-Za-z][A-Za-z .]{2,45})",
        joined,
        re.IGNORECASE,
    )
    if bname_m:
        receiver_name = re.sub(r"\s+", " ", bname_m.group(1)).strip()

    acc_num_m = re.search(r"(?:Account\s*Number)[\s:.\-]*([XK*0-9]{6,22})", joined, re.IGNORECASE)
    if acc_num_m:
        receiver_account = normalize_masked_account(acc_num_m.group(1))

    bank_label_m = re.search(r"Bank\s*Name[\s:.\-]+([A-Za-z][A-Za-z &.\-]{2,45})", joined, re.IGNORECASE)
    if bank_label_m:
        receiver_bank = re.sub(r"\s+", " ", bank_label_m.group(1)).strip()

    # 8b. Parse 'Paid to' / 'To' / 'Beneficiary details' block specifically
    to_block = ""
    to_m = re.search(
        r"(?:^|\n)\s*(?:Paid\s*to|Sent\s*to|Beneficiary\s*details|To)\s*[:\-]?\s*\n([\s\S]*?)(?=(?:\n\s*(?:Banking\s*Name|Transfer\s*Details|PhonePe\s*Transaction|Transaction\s*ID|Debited\s*from|From\b|UTR\b|Powered\s*by|Account\s*Summary))|\Z)",
        joined,
        re.IGNORECASE,
    )
    if to_m:
        to_block = to_m.group(1).strip()

    if to_block:
        # Also check multi-line IFSC inside to_block (e.g. 'IFSC\nFDRLOO01088')
        ifsc_block_m = re.search(r"\b([A-Z]{4}[0O][A-Z0-9O]{6})\b", to_block)
        if ifsc_block_m:
            receiver_ifsc = clean_ifsc(ifsc_block_m.group(1))

        to_lines = [l.strip() for l in to_block.splitlines() if l.strip()]
        for l in to_lines:
            cleaned_l = re.sub(r"^[^A-Za-z0-9XK*]+", "", l).strip()
            cleaned_l = re.sub(
                r"\s+(?:[₹$]|Rs\.?|INR|AED)?\s*[0-9,]+(?:\.[0-9]{1,2})?\s*$",
                "",
                cleaned_l,
                flags=re.IGNORECASE,
            ).strip()

            # Check for Account Number inside To block
            acc_m = re.search(
                r"(?:A/C(?:\s*No\.?)?|Account\s*Number|Account)?[\s:\-]*([XK*]{4,}\s*[0-9]{3,8}|\b\d{9,18}\b)",
                cleaned_l,
                re.IGNORECASE,
            )
            if acc_m:
                cand_acc = normalize_masked_account(acc_m.group(1))
                if cand_acc != rrn and cand_acc != txn_id:
                    receiver_account = cand_acc

            # Check for UPI ID
            upi_m = re.search(r"([a-zA-Z0-9.\-_]+@[a-zA-Z0-9]+)", cleaned_l)
            if upi_m and receiver_account == "N/A":
                receiver_account = upi_m.group(1)

            # Check for Bank Name
            if re.search(r"\bBANK\b", cleaned_l, re.IGNORECASE):
                if not re.search(r"^Bank\s*Name$", cleaned_l, re.I):
                    receiver_bank = re.sub(r"^Bank\s*Name[\s:\-]*", "", cleaned_l, flags=re.IGNORECASE).strip()
                continue

            # Check for Beneficiary Name
            if receiver_name == "N/A":
                name_cand = re.sub(r"^(?:Account\s*Name|Name)[\s:\-]+", "", cleaned_l, flags=re.IGNORECASE).strip()
                if (
                    re.match(r"^[A-Za-z][A-Za-z .]{2,42}$", name_cand)
                    and not re.search(
                        r"\b(BANK|IFSC|IMPS|NEFT|UPI|ACCOUNT|NUMBER|NICKNAME|COUNTRY|ADDRESS|DUBAI|TRANSFER|SUCCESS|FDRL)\b",
                        name_cand,
                        re.IGNORECASE,
                    )
                ):
                    receiver_name = name_cand

    # 8c. Fallback scan for known bank names if receiver_bank is still N/A
    if receiver_bank == "N/A":
        for kb in KNOWN_BANKS:
            if re.search(r"\b" + re.escape(kb) + r"\b", joined, re.IGNORECASE):
                receiver_bank = kb
                break

    return {
        "filename": filename or "receipt.jpg",
        "status": status,
        "payment_mode": payment_mode,
        "currency": currency,
        "transaction_id": txn_id or "N/A",
        "rrn": rrn or "N/A",
        "date_time": date_time,
        "amount": f"{amount_val:,.2f}",
        "amount_numeric": amount_val,
        "charges": f"{charges_val:,.2f}",
        "charges_numeric": charges_val,
        "receiver_name": receiver_name,
        "receiver_account": receiver_account,
        "receiver_ifsc": receiver_ifsc,
        "receiver_bank": receiver_bank,
        "raw_text": joined,
    }


def extract_from_image(image_path: str) -> Dict[str, Any]:
    """
    Runs multi-pass Tesseract OCR (--psm 6 and --psm 4) on preprocessed variants
    and returns the most complete parsed transaction dictionary.
    """
    variants = preprocess_for_ocr(image_path)
    best_parsed: Optional[Dict[str, Any]] = None
    best_score = -1

    for var_img in variants[:2]:
        for psm in ("--oem 3 --psm 6", "--oem 3 --psm 4"):
            raw_text = pytesseract.image_to_string(var_img, config=psm)
            parsed = parse_receipt_text(raw_text, filename=os.path.basename(image_path))
            score = 0
            if parsed["amount_numeric"] > 0:
                score += 5
            if parsed["transaction_id"] != "N/A":
                score += 2
            if parsed["rrn"] != "N/A":
                score += 2
            if parsed["receiver_name"] != "N/A":
                score += 3
            if parsed["receiver_account"] != "N/A":
                score += 3
            if parsed["receiver_bank"] != "N/A":
                score += 2
            if score > best_score:
                best_score = score
                best_parsed = parsed

    assert best_parsed is not None
    best_parsed["image_path"] = os.path.abspath(image_path)
    return best_parsed


def extract_batch(image_paths: List[str]) -> List[Dict[str, Any]]:
    """Extracts structured data from multiple receipt images."""
    results = []
    for idx, path in enumerate(image_paths, start=1):
        rec = extract_from_image(path)
        rec["index"] = idx
        results.append(rec)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract OCR data from multiple transaction receipt images.")
    parser.add_argument("images", nargs="+", help="Paths to receipt image files")
    parser.add_argument("-o", "--output", help="Optional path to save extracted JSON")
    args = parser.parse_args()

    extracted = extract_batch(args.images)
    print(json.dumps(extracted, indent=2, ensure_ascii=False))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(extracted, f, indent=2, ensure_ascii=False)
