# -*- coding: utf-8 -*-
"""
@author: zkchen

figure 3B.xlsx
1. Read the GeneSymbols in the first column of 'figure 3B (total genes). xlsx'
2. Bridge the gene to the Targeted ID using GENENAME from 'P1-01-TTD_target_download.txt'
3. Use 'P1-07-Drug Target Mapping. xlsx' (Target ID, Drug ID, Highest_stus, MOA) as the matching benchmark
4. Use DrugID to match 'P1-02-TTD_drug_download.txt' to retrieve TRADNAME/DRUGTYPE/HIGHSTAT/HERCLAS/DRUGSHIL
5. Filtering (executed sequentially before the output line):
  (1) SMILES excludes NA and failed
  (2) Exclude rows containing Discontinued/Withdrawn/Terminated from Drug Status
  (3) MOA only retains a clear list of inhibitory/activating words (see MOA_INHIBIT/MOA_ACTIVATE)
6. Add MOA rename column (inhibition class ->inhibitor, activation class ->agonist) and Label column:
  up + Agonist -> Risk
  down + Inhibitor -> Risk
  up + Inhibitor -> Dual-effect
  down + Agonist -> Dual-effect
7. Output figure 3B.xlsx:
  GeneSymbol | Regulation | TargetID | DrugID | Highest_status | MOA |
  Drug Name | Drug Type | Drug Status | Therapeutic Class | SMILES | MOA-rename | Label
"""

import os
import sys
from collections import OrderedDict, Counter

import zipfile
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.dirname(os.path.abspath(__file__))
XLSX_IN = os.path.join(BASE, "figure 3B (total genes).xlsx")
P107 = os.path.join(BASE, "P1-07-Drug-TargetMapping.xlsx")
TARGETS_TXT = os.path.join(BASE, "P1-01-TTD_target_download.txt")
DRUGS_TXT = os.path.join(BASE, "P1-02-TTD_drug_download.txt")
XLSX_OUT = os.path.join(BASE, "figure 3B.xlsx")

NA = "NA"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

ALIAS_MAP = {"COX2": "PTGS2", "FAAH": "FAAH1"}


# --- xlsx
def read_xlsx_rows(path):
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall(f"{NS}si"):
            shared.append("".join(t.text or "" for t in si.iter(f"{NS}t")))
    sheet = None
    for name in z.namelist():
        if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"):
            sheet = name
            break
    root = ET.fromstring(z.read(sheet))
    rows = []
    for row in root.iter(f"{NS}row"):
        vals = []
        for c in row.findall(f"{NS}c"):
            t = c.get("t")
            v = c.find(f"{NS}v")
            is_el = c.find(f"{NS}is")
            if is_el is not None:  # inlineStr
                val = "".join(t2.text or "" for t2 in is_el.iter(f"{NS}t"))
            else:
                val = "" if v is None else (v.text or "")
                if t == "s" and val != "":
                    val = shared[int(val)]
            vals.append(val)
        rows.append(vals)
    return rows

def col_letter(idx):
    s = ""
    while idx > 0:
        idx, r = divmod(idx - 1, 26)
        s = chr(65 + r) + s
    return s

def write_xlsx(path, rows, header_bold=True, col_widths=None):
    n_rows = len(rows)
    n_cols = max(len(r) for r in rows)
    dim = f"A1:{col_letter(n_cols)}{n_rows}"

    cols_xml = ""
    if col_widths:
        parts = []
        for i, w in enumerate(col_widths, start=1):
            parts.append(f'<col min="{i}" max="{i}" width="{w}" customWidth="1"/>')
        cols_xml = "<cols>" + "".join(parts) + "</cols>"

    body = []
    for ri, row in enumerate(rows, start=1):
        cells = []
        for ci in range(1, n_cols + 1):
            val = row[ci - 1] if ci - 1 < len(row) else ""
            style = ' s="1"' if (header_bold and ri == 1) else ""
            cells.append(
                f'<c r="{col_letter(ci)}{ri}"{style} t="inlineStr">'
                f"<is><t xml:space=\"preserve\">{escape(val)}</t></is></c>"
            )
        body.append(f'<row r="{ri}">' + "".join(cells) + "</row>")
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="{dim}"/>'
        '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        f"{cols_xml}"
        "<sheetData>" + "".join(body) + "</sheetData>"
        "</worksheet>"
    )

    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
        "<font><b/><sz val=\"11\"/><name val=\"Calibri\"/></font></fonts>"
        '<fills count="2"><fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        "</styleSheet>"
    )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        "</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets><sheet name=\"Sheet1\" sheetId=\"1\" r:id=\"rId1\"/></sheets>"
        "</workbook>"
    )
    wb_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        "</Relationships>"
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("xl/workbook.xml", workbook_xml)
        z.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        z.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        z.writestr("xl/styles.xml", styles_xml)

def parse_drugs(path):
    """{drug_id: {field: value}}"""
    drugs = {}
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").rstrip("\r").split("\t")
            if len(parts) < 3:
                continue
            did, field = parts[0].strip(), parts[1].strip()
            if field == "DRUG__ID":
                continue
            value = "\t".join(parts[2:]).strip()
            drugs.setdefault(did, {})[field] = value
    return drugs

# --- Filtering rules
# MOA Clearly suppress word list -> MOA-rename = Inhibitor
MOA_INHIBIT = {
    "Inhibitor",
    "Inhibitor (gating inhibitor)",
    "Antagonist",
    "Blocker",
    "Blocker (channel blocker)",
    "Suppressor",
    "Antisense",
}
# MOA Clear activation word list -> MOA-rename = Agonist
MOA_ACTIVATE = {
    "Agonist",
    "Activator",
    "Stimulator",
    "Regulator (upregulator)",
    "Mimetic",
}
MOA_KEEP = MOA_INHIBIT | MOA_ACTIVATE

# Drug Status Exclude keywords (case insensitive substring matching)
STATUS_EXCLUDE = ("discontinued", "withdrawn", "terminated")


def parse_targetid_genes(path):
    """TargetID -> GENENAME Original string (supports multiple genes' A; B ')"""
    tid2genes = OrderedDict()
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            p = line.rstrip("\n").rstrip("\r").split("\t")
            if len(p) >= 3 and p[1] == "GENENAME":
                tid2genes[p[0]] = p[2].strip()
    return tid2genes


def parse_druginfo_names(path):
    """(TargetID, DrugID) -> Drug Name（TTD Targets DRUGINFO，
    INN/Code name, as a fallback when there is no product name）"""
    pair2name = {}
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            p = line.rstrip("\n").rstrip("\r").split("\t")
            if len(p) >= 4 and p[1] == "DRUGINFO":
                rest = [x.strip() for x in p[2:] if x.strip()]
                if len(rest) >= 2:
                    pair2name.setdefault((p[0], rest[0]), rest[1])
    return pair2name


def main():
    # --- INPUT
    fig_rows = read_xlsx_rows(XLSX_IN)
    header = fig_rows[0]
    data = [r for r in fig_rows[1:] if any(c.strip() for c in r)]

    p107_rows = read_xlsx_rows(P107)[1:]
    p107_rows = [r for r in p107_rows if r and r[0].strip()]

    # --- TargetID -> Genes (bridged by TTD Targets GENENAME)
    tid2genes = parse_targetid_genes(TARGETS_TXT)
    gene2tids = {}
    for tid, gname in tid2genes.items():
        for g in gname.replace(",", ";").split(";"):
            g = g.strip().upper()
            if g:
                gene2tids.setdefault(g, []).append(tid)

    # --- Gene -> P1-07 mapping line index (maintaining the order within the P1-07 file)
    gene2maps = {}
    no_gene_tid = 0
    for r in p107_rows:
        tid, did, hstat, moa = r[0].strip(), r[1].strip(), r[2].strip(), r[3].strip()
        gname = tid2genes.get(tid)
        if not gname:
            no_gene_tid += 1
            continue
        for g in gname.replace(",", ";").split(";"):
            g = g.strip().upper()
            if g:
                gene2maps.setdefault(g, []).append((tid, did, hstat, moa))

    # --- TTD Drugs
    drugs_db = parse_drugs(DRUGS_TXT)
    pair2name = parse_druginfo_names(TARGETS_TXT)

    # --- Generate
    new_cols = ["TargetID", "DrugID", "Highest_status", "MOA",
                "Drug Name", "Drug Type", "Drug Status", "Therapeutic Class", "SMILES",
                "MOA-rename", "Label"]
    out_rows = [list(header) + new_cols]

    n_mapped = 0
    n_skipped = 0
    skipped_genes = []
    n_out = 0
    drop_counter = Counter()   # Each filtering rule eliminates the number of rows
    moa_excluded = Counter()   # Distribution of excluded values from MOA vocabulary
    na_counter = Counter()
    label_counter = Counter()
    rename_counter = Counter()
    alias_note = {}
    reg_anomaly = 0

    for r in data:
        gene = (r[0] if r else "").strip()
        key = ALIAS_MAP.get(gene.upper(), gene.upper())
        if key != gene.upper():
            alias_note[gene] = key
        base = list(r) + [""] * (len(header) - len(r)) if len(r) < len(header) else list(r)
        maps = gene2maps.get(key, [])
        if not maps:
            n_skipped += 1
            skipped_genes.append(gene)
            continue
        n_mapped += 1
        for tid, did, hstat, moa in maps:
            d = drugs_db.get(did, {})
            moa_v = NA if moa in ("", ".") else moa
            # Drug Name: TRADNAME(Product name) priority
            name = d.get("TRADNAME", "") or pair2name.get((tid, did), "") or NA
            dtype = d.get("DRUGTYPE", "") or NA
            status = d.get("HIGHSTAT", "") or hstat or NA
            tclas = d.get("THERCLAS", "") or NA
            smiles = d.get("DRUGSMIL", "") or NA

            # --- Filter (1) SMILES to exclude NA/failed
            s = smiles.strip()
            if s == NA or "failed" in s.lower():
                drop_counter["SMILES(NA/failed)"] += 1
                continue
            # --- Filter (2) Drug Status Exclude Discontinued/Withdrawn/Terminated
            st = status.lower()
            if any(kw in st for kw in STATUS_EXCLUDE):
                drop_counter["Drug Status(Discontinued/Withdrawn/Terminated)"] += 1
                continue
            # --- Filter (3) MOA retains only the explicitly inhibitory/activatory lexicon
            if moa_v not in MOA_KEEP:
                drop_counter["MOA(Non-specific inhibition/activation)"] += 1
                moa_excluded[moa_v] += 1
                continue

            # --- MOA-rename
            if moa_v in MOA_INHIBIT:
                moa_rename = "Inhibitor"
            else:
                moa_rename = "Agonist"
            rename_counter[moa_rename] += 1

            # --- Label
            reg = base[1].strip().lower() if len(base) > 1 else ""
            if reg not in ("up", "down"):
                reg_anomaly += 1
                continue
            if reg == "up":
                label = "Risk" if moa_rename == "Agonist" else "Dual-effect"
            else:  # down
                label = "Risk" if moa_rename == "Inhibitor" else "Dual-effect"
            label_counter[label] += 1

            for k, v in [("Drug Name", name), ("Drug Type", dtype), ("Drug Status", status),
                         ("Therapeutic Class", tclas), ("SMILES", smiles)]:
                if v == NA:
                    na_counter[k] += 1

            out_rows.append(base + [tid, did, hstat, moa_v, name, dtype, status, tclas, smiles,
                                    moa_rename, label])
            n_out += 1

    write_xlsx(XLSX_OUT, out_rows, header_bold=True,
               col_widths=[14, 12, 11, 11, 15, 13, 24, 20, 14, 26, 60, 12, 12])

    print("\n================================")
    print(f"There are genes mapped to P1-07: {n_mapped}/{len(data)}")
    print(f"Excluded genes without mapping: {n_skipped}")
    print(f"Number of mapped rows before filtering: {sum(drop_counter.values()) + n_out}")
    print("Each filtering rule eliminates the number of rows:")
    for k, v in drop_counter.items():
        print(f"  {k}: {v}")
    if moa_excluded:
        print("  Details of excluded values from MOA vocabulary:", dict(moa_excluded.most_common()))
    print(f"Final output line count (excluding header): {n_out}")
    print(f"Number of genes involved: {len(set(r[0] for r in out_rows[1:]))}")
    print("MOA rename distribution:", dict(rename_counter))
    print("Label distribution:", dict(label_counter))
    print("NA count for each field:", dict(na_counter))
    if reg_anomaly:
        print(f"!! Regulation skipped lines that are not up/down: {reg_anomaly}")
    if alias_note:
        print("Alias:", alias_note)
    print(f"\n[output] {XLSX_OUT}")


if __name__ == "__main__":
    main()
