"""Generate the WSDM account-level schematic figures.

This script intentionally uses only the Python standard library.  It writes
vector PDFs directly so the paper figures do not depend on matplotlib, numpy,
TikZ, or a local TeX installation.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path


BETA = 0.20
NET_RATE = 0.18
LAMBDA_HARM = 0.85
LAMBDA_NET = 0.45


RGB = tuple[float, float, float]


INK: RGB = (0.15, 0.20, 0.22)
MUTED: RGB = (0.42, 0.47, 0.49)
GRID: RGB = (0.86, 0.89, 0.89)
VALUE_PASS: RGB = (0.82, 0.91, 0.88)
VALUE_BLOCK: RGB = (0.46, 0.70, 0.65)
LOSS_PASS: RGB = (0.93, 0.82, 0.79)
LOSS_BLOCK: RGB = (0.77, 0.35, 0.29)
INDIGO: RGB = (0.24, 0.31, 0.58)
RUST: RGB = (0.68, 0.30, 0.22)
COPPER: RGB = (0.67, 0.36, 0.20)
AMBER: RGB = (0.72, 0.50, 0.12)
ORANGE: RGB = (0.95, 0.66, 0.08)
VIOLET: RGB = (0.45, 0.33, 0.62)
SAND: RGB = (0.97, 0.92, 0.82)
WHITE: RGB = (1.0, 1.0, 1.0)
VALUE_GREY: RGB = (0.36, 0.39, 0.40)
LOSS_RED: RGB = (0.72, 0.24, 0.20)
GROUP_OUTLINE_GROUP = 1
GROUP_OUTLINE: RGB = ORANGE
GROUP_OUTLINE_WIDTH = 0.75
GROUP_OUTLINE_PAD = 0.65


@dataclass(frozen=True)
class Account:
    key: str
    value: float
    loss: float
    group: int


@dataclass(frozen=True)
class FigureData:
    accounts: tuple[Account, ...]
    value_loss_order: tuple[int, ...]
    priced_order: tuple[int, ...]
    value_loss_blocked: frozenset[int]
    harm_blocked: frozenset[int]
    net_blocked: frozenset[int]
    priced_blocked: frozenset[int]
    budget: float


def _pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _num(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _rgb(rgb: RGB) -> str:
    return " ".join(_num(component) for component in rgb)


class Canvas:
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height
        self.commands: list[str] = []

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: RGB | None = None,
        stroke: RGB | None = None,
        line_width: float = 0.7,
        dash: tuple[float, ...] | None = None,
    ) -> None:
        self.commands.append("q")
        if fill is not None:
            self.commands.append(f"{_rgb(fill)} rg")
        if stroke is not None:
            self.commands.append(f"{_rgb(stroke)} RG")
            self.commands.append(f"{_num(line_width)} w")
            if dash is not None:
                self.commands.append(f"[{' '.join(_num(v) for v in dash)}] 0 d")
        self.commands.append(
            f"{_num(x)} {_num(y)} {_num(width)} {_num(height)} re "
            + ("B" if fill is not None and stroke is not None else "f" if fill is not None else "S")
        )
        self.commands.append("Q")

    def line(
        self,
        points: list[tuple[float, float]],
        *,
        stroke: RGB = INK,
        line_width: float = 0.8,
        dash: tuple[float, ...] | None = None,
    ) -> None:
        if len(points) < 2:
            return
        self.commands.append("q")
        self.commands.append(f"{_rgb(stroke)} RG")
        self.commands.append(f"{_num(line_width)} w")
        if dash is not None:
            self.commands.append(f"[{' '.join(_num(v) for v in dash)}] 0 d")
        x0, y0 = points[0]
        self.commands.append(f"{_num(x0)} {_num(y0)} m")
        for x, y in points[1:]:
            self.commands.append(f"{_num(x)} {_num(y)} l")
        self.commands.append("S")
        self.commands.append("Q")

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        *,
        fill: RGB,
        stroke: RGB | None = None,
        line_width: float = 0.5,
    ) -> None:
        k = 0.5522847498 * radius
        self.commands.append("q")
        self.commands.append(f"{_rgb(fill)} rg")
        if stroke is not None:
            self.commands.append(f"{_rgb(stroke)} RG")
            self.commands.append(f"{_num(line_width)} w")
        self.commands.append(f"{_num(x + radius)} {_num(y)} m")
        self.commands.append(f"{_num(x + radius)} {_num(y + k)} {_num(x + k)} {_num(y + radius)} {_num(x)} {_num(y + radius)} c")
        self.commands.append(f"{_num(x - k)} {_num(y + radius)} {_num(x - radius)} {_num(y + k)} {_num(x - radius)} {_num(y)} c")
        self.commands.append(f"{_num(x - radius)} {_num(y - k)} {_num(x - k)} {_num(y - radius)} {_num(x)} {_num(y - radius)} c")
        self.commands.append(f"{_num(x + k)} {_num(y - radius)} {_num(x + radius)} {_num(y - k)} {_num(x + radius)} {_num(y)} c")
        self.commands.append("B" if stroke is not None else "f")
        self.commands.append("Q")

    def text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        size: float = 8.0,
        color: RGB = INK,
        font: str = "F1",
        align: str = "left",
    ) -> None:
        width = len(text) * size * 0.50
        if align == "center":
            x -= width / 2
        elif align == "right":
            x -= width
        self.commands.append("q")
        self.commands.append(f"{_rgb(color)} rg")
        self.commands.append(f"BT /{font} {_num(size)} Tf {_num(x)} {_num(y)} Td ({_pdf_text(text)}) Tj ET")
        self.commands.append("Q")

    def rotated_text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        angle_degrees: float,
        size: float = 8.0,
        color: RGB = INK,
        font: str = "F1",
        align: str = "left",
    ) -> None:
        width = len(text) * size * 0.50
        offset = 0.0
        if align == "center":
            offset = -width / 2
        elif align == "right":
            offset = -width
        angle = math.radians(angle_degrees)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        tx = x + offset * cos_angle
        ty = y + offset * sin_angle
        self.commands.append("q")
        self.commands.append(f"{_rgb(color)} rg")
        self.commands.append(
            f"BT /{font} {_num(size)} Tf "
            f"{_num(cos_angle)} {_num(sin_angle)} "
            f"{_num(-sin_angle)} {_num(cos_angle)} "
            f"{_num(tx)} {_num(ty)} Tm ({_pdf_text(text)}) Tj ET"
        )
        self.commands.append("Q")

    def content(self) -> bytes:
        return ("\n".join(self.commands) + "\n").encode("latin-1")


class PDF:
    def __init__(self) -> None:
        self.pages: list[Canvas] = []

    def add_page(self, canvas: Canvas) -> None:
        self.pages.append(canvas)

    def write(self, path: Path) -> None:
        objects: list[bytes] = [b""]

        def add(obj: bytes) -> int:
            objects.append(obj)
            return len(objects) - 1

        catalog_id = add(b"")
        pages_id = add(b"")
        font_regular_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        font_bold_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
        page_ids: list[int] = []
        for canvas in self.pages:
            stream = canvas.content()
            content_id = add(
                b"<< /Length "
                + str(len(stream)).encode("ascii")
                + b" >>\nstream\n"
                + stream
                + b"endstream"
            )
            page_id = add(
                (
                    f"<< /Type /Page /Parent {pages_id} 0 R "
                    f"/MediaBox [0 0 {_num(canvas.width)} {_num(canvas.height)}] "
                    f"/Resources << /Font << /F1 {font_regular_id} 0 R /F2 {font_bold_id} 0 R >> >> "
                    f"/Contents {content_id} 0 R >>"
                ).encode("ascii")
            )
            page_ids.append(page_id)

        objects[catalog_id] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("ascii")
        kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
        objects[pages_id] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")

        output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for obj_id, obj in enumerate(objects[1:], start=1):
            offsets.append(len(output))
            output.extend(f"{obj_id} 0 obj\n".encode("ascii"))
            output.extend(obj)
            output.extend(b"\nendobj\n")
        xref = len(output)
        output.extend(f"xref\n0 {len(objects)}\n".encode("ascii"))
        output.extend(b"0000000000 65535 f\n")
        for offset in offsets[1:]:
            output.extend(f"{offset:010d} 00000 n\n".encode("ascii"))
        output.extend(
            (
                f"trailer\n<< /Size {len(objects)} /Root {catalog_id} 0 R >>\n"
                f"startxref\n{xref}\n%%EOF\n"
            ).encode("ascii")
        )
        path.write_bytes(output)


def group_has_outline(group: int) -> bool:
    return group == GROUP_OUTLINE_GROUP


def draw_group_outline(
    canvas: Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    line_width: float = GROUP_OUTLINE_WIDTH,
    pad: float = GROUP_OUTLINE_PAD,
    dash: tuple[float, ...] | None = None,
    include_bottom: bool = True,
) -> None:
    if include_bottom:
        left = x - pad
        right = x + width + pad
        bottom = y - pad
        top = y + height + pad
        canvas.rect(
            left,
            bottom,
            right - left,
            top - bottom,
            stroke=GROUP_OUTLINE,
            line_width=line_width,
            dash=dash,
        )
        return
    left = x - pad
    right = x + width + pad
    bottom = y
    top = y + height + pad
    canvas.line(
        [(left, bottom), (left, top), (right, top), (right, bottom)],
        stroke=GROUP_OUTLINE,
        line_width=line_width,
        dash=dash,
    )


def accounts() -> tuple[Account, ...]:
    rng = random.Random(19)
    table: list[Account] = []
    for idx in range(54):
        group = 1 if rng.random() < 0.48 else 0
        loss = rng.uniform(1.05, 3.75) + (0.15 if group else 0.0)
        center = 3.55 if group == 0 else 2.30
        score = max(0.45, rng.lognormvariate(math.log(center), 0.34))
        value = loss * score
        table.append(Account(f"a{idx + 1:02d}", value, loss, group))
    return tuple(table)


def value_loss_score(account: Account) -> float:
    if account.loss <= 0:
        return float("inf")
    return account.value / account.loss


def harm_score(account: Account) -> float:
    return account.loss


def net_score(account: Account) -> float:
    return account.loss - NET_RATE * account.value


def sorted_indices(scores: list[float], *, reverse: bool) -> tuple[int, ...]:
    return tuple(sorted(range(len(scores)), key=scores.__getitem__, reverse=reverse))


def budget_prefix(table: tuple[Account, ...], order: tuple[int, ...], budget: float) -> frozenset[int]:
    spent = 0.0
    chosen: list[int] = []
    for index in order:
        spent += table[index].value
        chosen.append(index)
        if spent >= budget:
            break
    return frozenset(chosen)


def budget_suffix(table: tuple[Account, ...], order: tuple[int, ...], budget: float) -> frozenset[int]:
    spent = 0.0
    chosen: list[int] = []
    for index in reversed(order):
        spent += table[index].value
        chosen.append(index)
        if spent >= budget:
            break
    return frozenset(chosen)


def liability(data: FigureData, index: int) -> float:
    amount = 0.0
    if index not in data.harm_blocked:
        amount += LAMBDA_HARM
    if index not in data.net_blocked:
        amount += LAMBDA_NET
    return amount


def priced_loss(data: FigureData, index: int) -> float:
    return max(data.accounts[index].loss - liability(data, index), 0.0)


def priced_score(data: FigureData, index: int) -> float:
    loss = priced_loss(data, index)
    if loss <= 0:
        return float("inf")
    return data.accounts[index].value / loss


def objection_flags(data: FigureData, index: int) -> tuple[bool, bool]:
    if index not in data.value_loss_blocked:
        return (False, False)
    return (index not in data.harm_blocked, index not in data.net_blocked)


def objected(data: FigureData, index: int) -> bool:
    return any(objection_flags(data, index))


def would_object_if_blocked(data: FigureData, index: int) -> bool:
    return index not in data.harm_blocked or index not in data.net_blocked


def build_data() -> FigureData:
    table = accounts()
    budget = BETA * sum(account.value for account in table)
    value_loss_order = sorted_indices([value_loss_score(account) for account in table], reverse=True)
    harm_order = sorted_indices([harm_score(account) for account in table], reverse=True)
    net_order = sorted_indices([net_score(account) for account in table], reverse=True)
    value_loss_blocked = budget_suffix(table, value_loss_order, budget)
    harm_blocked = budget_prefix(table, harm_order, budget)
    net_blocked = budget_prefix(table, net_order, budget)
    partial = FigureData(
        accounts=table,
        value_loss_order=value_loss_order,
        priced_order=value_loss_order,
        value_loss_blocked=value_loss_blocked,
        harm_blocked=harm_blocked,
        net_blocked=net_blocked,
        priced_blocked=frozenset(),
        budget=budget,
    )
    priced_order = sorted_indices([priced_score(partial, index) for index in range(len(table))], reverse=True)
    priced_blocked = budget_suffix(table, priced_order, budget)
    data = FigureData(
        accounts=table,
        value_loss_order=value_loss_order,
        priced_order=priced_order,
        value_loss_blocked=value_loss_blocked,
        harm_blocked=harm_blocked,
        net_blocked=net_blocked,
        priced_blocked=priced_blocked,
        budget=budget,
    )
    assert data.value_loss_order == sorted_indices([value_loss_score(account) for account in table], reverse=True)
    assert data.priced_order == sorted_indices([priced_score(data, index) for index in range(len(table))], reverse=True)
    assert any(objected(data, index) for index in data.value_loss_blocked)
    assert any(index in data.value_loss_blocked and index not in data.priced_blocked for index in range(len(table)))
    assert any(index not in data.value_loss_blocked and index in data.priced_blocked for index in range(len(table)))
    return data


def suffix_cut(order: tuple[int, ...], blocked: frozenset[int]) -> int:
    positions = [pos for pos, index in enumerate(order) if index in blocked]
    return min(positions) if positions else len(order)


def draw_blocked_cut(
    canvas: Canvas,
    order: tuple[int, ...],
    blocked: frozenset[int],
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    label: str | None,
    side_labels: bool = True,
    fill: RGB | None = SAND,
    stroke: RGB = AMBER,
    dash: tuple[float, ...] | None = (3.5, 2.0),
) -> float:
    cut = suffix_cut(order, blocked)
    step = width / len(order)
    cut_x = x + cut * step
    if fill is not None:
        canvas.rect(cut_x, y - 4, x + width - cut_x, height + 11, fill=fill)
    canvas.line([(cut_x, y - 4), (cut_x, y + height + 7)], stroke=stroke, line_width=1.1, dash=dash)
    if label is not None:
        canvas.text(cut_x + 3, y + height + 5, label, size=7.5, color=stroke, font="F2")
    if side_labels:
        canvas.text(x + width - 2, y + height + 5, "blocked suffix", size=7.0, color=INK, align="right")
        canvas.text(cut_x - 3, y + height + 5, "passed prefix", size=7.0, color=MUTED, align="right")
    return cut_x


def draw_account_bars(
    canvas: Canvas,
    data: FigureData,
    order: tuple[int, ...],
    x: float,
    y: float,
    width: float,
    height: float,
    blocked: frozenset[int],
    *,
    adjusted: bool,
    mark_objections: bool,
    value_pass: RGB = VALUE_PASS,
    value_block: RGB = VALUE_BLOCK,
    loss_pass: RGB = LOSS_PASS,
    loss_block: RGB = LOSS_BLOCK,
    show_groups: bool = True,
    highlight_labels: dict[int, str] | None = None,
    penalty_marks: frozenset[int] | None = None,
) -> None:
    highlight_labels = highlight_labels or {}
    max_total = max(account.value + account.loss for account in data.accounts)
    scale = height / max_total
    step = width / len(order)
    bar_w = step * 0.62
    for pos, index in enumerate(order):
        account = data.accounts[index]
        bx = x + pos * step + (step - bar_w) / 2
        value_h = account.value * scale
        loss = priced_loss(data, index) if adjusted else account.loss
        loss_h = loss * scale
        original_loss_h = account.loss * scale
        is_blocked = index in blocked
        canvas.rect(bx, y, bar_w, value_h, fill=value_block if is_blocked else value_pass)
        canvas.rect(bx, y + value_h, bar_w, loss_h, fill=loss_block if is_blocked else loss_pass)
        if show_groups and group_has_outline(account.group):
            draw_group_outline(canvas, bx, y, bar_w, value_h + original_loss_h, include_bottom=False)
        harm_pass, net_pass = objection_flags(data, index)
        if mark_objections and index in data.value_loss_blocked and (harm_pass or net_pass):
            canvas.rect(
                bx - 0.8,
                y - 0.8,
                bar_w + 1.6,
                value_h + original_loss_h + 1.6,
                stroke=COPPER,
                line_width=0.9,
            )
            dot_y = y + value_h + original_loss_h + 3.2
            if harm_pass:
                canvas.circle(bx + bar_w * 0.35, dot_y, 1.4, fill=AMBER)
            if net_pass:
                canvas.circle(bx + bar_w * 0.65, dot_y, 1.4, fill=VIOLET)
        if index in highlight_labels:
            top = y + value_h + original_loss_h
            canvas.rect(bx - 1.4, y - 1.4, bar_w + 2.8, value_h + original_loss_h + 2.8, stroke=INK, line_width=1.05)
            label = highlight_labels[index]
            draw_account_marker(canvas, bx + bar_w / 2, top + 8.0, label)
        if adjusted and liability(data, index) > 0 and (penalty_marks is None or index in penalty_marks):
            original_top = y + value_h + original_loss_h
            adjusted_top = y + value_h + loss_h
            canvas.line([(bx, original_top), (bx + bar_w, original_top)], stroke=COPPER, line_width=0.45, dash=(2.4, 1.7))
            if (mark_objections and objected(data, index)) or (penalty_marks is not None and index in penalty_marks):
                canvas.line([(bx + bar_w / 2, original_top - 0.5), (bx + bar_w / 2, adjusted_top + 0.5)], stroke=COPPER, line_width=0.45)


def draw_common_legend(
    canvas: Canvas,
    x: float,
    y: float,
    *,
    include_curves: bool,
    value_color: RGB = VALUE_BLOCK,
    loss_color: RGB = LOSS_BLOCK,
) -> None:
    legend_w = 202 if include_curves else 116
    canvas.rect(x, y, legend_w, 30, fill=WHITE, stroke=GRID, line_width=0.6)
    canvas.rect(x + 6, y + 20, 9, 5, fill=value_color)
    canvas.text(x + 19, y + 20.5, "value", size=7, color=INK)
    canvas.rect(x + 60, y + 20, 9, 5, fill=loss_color)
    canvas.text(x + 73, y + 20.5, "loss", size=7, color=INK)
    canvas.rect(x + 6, y + 10, 9, 5, fill=WHITE, stroke=GRID, line_width=0.45)
    canvas.text(x + 19, y + 10.5, "G0", size=7, color=INK)
    draw_group_outline(canvas, x + 60, y + 10, 9, 5, line_width=0.6, pad=0.0)
    canvas.text(x + 73, y + 10.5, "G1 outline", size=7, color=INK)
    if include_curves:
        canvas.line([(x + 113, y + 22.5), (x + 130, y + 22.5)], stroke=INK, line_width=1.1)
        canvas.text(x + 134, y + 20.5, "blocked loss share", size=7, color=INK)
        canvas.line([(x + 113, y + 12.5), (x + 130, y + 12.5)], stroke=MUTED, line_width=1.0, dash=(3.0, 2.0))
        canvas.text(x + 134, y + 10.5, "blocked value share", size=7, color=INK)


def draw_group_block_summary(canvas: Canvas, data: FigureData, x: float, y: float) -> None:
    canvas.rect(x, y, 177, 30, fill=WHITE, stroke=GRID, line_width=0.6)
    canvas.text(x + 7, y + 20.5, "blocked by common cut", size=7.1, color=INK, font="F2")
    for col, group in enumerate((0, 1)):
        members = [index for index, account in enumerate(data.accounts) if account.group == group]
        blocked = sum(index in data.value_loss_blocked for index in members)
        group_x = x + 8 + col * 79
        if group == 0:
            canvas.rect(group_x, y + 8.5, 8, 5.5, fill=WHITE, stroke=GRID, line_width=0.45)
        else:
            draw_group_outline(canvas, group_x, y + 8.5, 8, 5.5, line_width=0.6, pad=0.0)
        canvas.text(group_x + 13, y + 7.8, f"G{group}: {blocked}/{len(members)}", size=7.3, color=INK)


def same_loss_example_pair(data: FigureData) -> tuple[int, int]:
    mean_loss = sum(account.loss for account in data.accounts) / len(data.accounts)
    max_loss_gap = 0.05 * mean_loss
    passed = frozenset(range(len(data.accounts))) - data.value_loss_blocked
    candidates: list[tuple[float, float, int, int]] = []
    for high_value_index in passed:
        high_value_account = data.accounts[high_value_index]
        for low_value_index in data.value_loss_blocked - data.priced_blocked:
            low_value_account = data.accounts[low_value_index]
            loss_gap = abs(high_value_account.loss - low_value_account.loss)
            value_gap = high_value_account.value - low_value_account.value
            if loss_gap <= max_loss_gap and value_gap > 0:
                candidates.append((value_gap, -loss_gap, high_value_index, low_value_index))
    if not candidates:
        raise RuntimeError("no same-loss passed/blocked example pair")
    _, _, high_value_index, low_value_index = max(candidates)
    return high_value_index, low_value_index


def example_labels(data: FigureData) -> dict[int, str]:
    high_value_index, low_value_index = same_loss_example_pair(data)
    return {high_value_index: "1", low_value_index: "2"}


def draw_yield_ranking_panel(
    canvas: Canvas,
    data: FigureData,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    show_groups: bool,
    highlight_labels: dict[int, str] | None,
    show_legend: bool,
    legend_x: float | None = None,
    legend_y: float | None = None,
    show_cut_labels: bool = False,
    shade_blocked: bool = False,
) -> None:
    for frac in (0.25, 0.50, 0.75, 1.00):
        gy = y + height * frac
        canvas.line([(x, gy), (x + width, gy)], stroke=GRID, line_width=0.45)
    draw_blocked_cut(
        canvas,
        data.value_loss_order,
        data.value_loss_blocked,
        x,
        y,
        width,
        height,
        label="cut" if show_cut_labels else None,
        side_labels=show_cut_labels,
        fill=SAND if shade_blocked else None,
        stroke=INK,
        dash=None,
    )
    draw_account_bars(
        canvas,
        data,
        data.value_loss_order,
        x,
        y,
        width,
        height * 0.78,
        data.value_loss_blocked,
        adjusted=False,
        mark_objections=False,
        value_pass=VALUE_GREY,
        value_block=VALUE_GREY,
        loss_pass=LOSS_RED,
        loss_block=LOSS_RED,
        show_groups=show_groups,
        highlight_labels=highlight_labels,
    )
    total_value = sum(account.value for account in data.accounts)
    total_loss = sum(account.loss for account in data.accounts)
    suffix_value = total_value
    suffix_loss = total_loss
    step = width / len(data.value_loss_order)
    value_points = [(x, y + height * suffix_value / total_value)]
    loss_points = [(x, y + height * suffix_loss / total_loss)]
    for pos, index in enumerate(data.value_loss_order, start=1):
        suffix_value -= data.accounts[index].value
        suffix_loss -= data.accounts[index].loss
        px = x + pos * step
        value_points.append((px, y + height * suffix_value / total_value))
        loss_points.append((px, y + height * suffix_loss / total_loss))
    canvas.line(loss_points, stroke=INK, line_width=1.25)
    canvas.line(value_points, stroke=MUTED, line_width=1.1, dash=(4.0, 2.5))
    canvas.line([(x, y), (x + width, y), (x + width, y + height), (x, y + height), (x, y)], stroke=INK, line_width=0.65)
    canvas.text(x + width / 2, y - 28, "accounts ordered by value per loss, high to low", size=9, color=INK, align="center")
    canvas.rotated_text(x - 23, y + height / 2, "value and loss per account", angle_degrees=90, size=8, color=INK, align="center")
    canvas.rotated_text(x + width + 22, y + height / 2, "blocked share", angle_degrees=-90, size=8, color=INK, align="center")
    if show_legend:
        draw_common_legend(
            canvas,
            legend_x if legend_x is not None else x + width - 133,
            legend_y if legend_y is not None else y + height + 23,
            include_curves=True,
            value_color=VALUE_GREY,
            loss_color=LOSS_RED,
        )


def figure_one(data: FigureData, path: Path) -> None:
    canvas = Canvas(480, 305)
    draw_yield_ranking_panel(
        canvas,
        data,
        46,
        60,
        362,
        172,
        show_groups=True,
        highlight_labels=example_labels(data),
        show_legend=True,
        legend_x=275,
        legend_y=255,
        show_cut_labels=True,
        shade_blocked=True,
    )
    draw_group_block_summary(canvas, data, 46, 255)
    pdf = PDF()
    pdf.add_page(canvas)
    pdf.write(path)


def panel_title(canvas: Canvas, x: float, y: float, letter: str, title: str) -> None:
    canvas.rect(x, y - 2, 15, 15, fill=INK)
    canvas.text(x + 7.5, y + 2, letter, size=9, color=WHITE, font="F2", align="center")
    canvas.text(x + 20, y + 2, title, size=9.5, color=INK, font="F2")


def draw_account_marker(canvas: Canvas, x: float, y: float, label: str) -> None:
    canvas.circle(x, y, 4.8, fill=WHITE, stroke=INK, line_width=0.75)
    canvas.text(x, y - 2.1, label, size=6.7, color=INK, font="F2", align="center")


def draw_small_badge(canvas: Canvas, x: float, y: float, label: str) -> None:
    draw_account_marker(canvas, x, y, label)


def draw_penalty_comparison(
    canvas: Canvas,
    data: FigureData,
    index: int,
    x: float,
    y: float,
    scale: float,
    *,
    label: str,
    status: str,
) -> None:
    account = data.accounts[index]
    bar_w = 15.0
    raw_x = x + 17
    priced_x = x + 54
    value_h = account.value * scale
    loss_h = account.loss * scale
    priced_loss_h = priced_loss(data, index) * scale

    draw_small_badge(canvas, x + 6, y + 76, label)
    canvas.text(x + 17, y + 73, status, size=7.4, color=INK, font="F2")
    for bx, shown_loss_h in ((raw_x, loss_h), (priced_x, priced_loss_h)):
        canvas.rect(bx, y, bar_w, value_h, fill=VALUE_GREY)
        canvas.rect(bx, y + value_h, bar_w, shown_loss_h, fill=LOSS_RED)
        canvas.rect(bx, y, bar_w, value_h + shown_loss_h, stroke=INK, line_width=0.45)
    canvas.text(raw_x + bar_w / 2, y - 12, "raw", size=6.8, color=MUTED, align="center")
    canvas.text(priced_x + bar_w / 2, y - 12, "priced", size=6.8, color=MUTED, align="center")
    canvas.text(
        x + 50,
        y - 24,
        f"score {value_loss_score(account):.2f} -> {priced_score(data, index):.2f}",
        size=6.4,
        color=INK,
        align="center",
    )

    if liability(data, index) > 0:
        original_top = y + value_h + loss_h
        adjusted_top = y + value_h + priced_loss_h
        canvas.line([(priced_x - 1, original_top), (priced_x + bar_w + 1, original_top)], stroke=COPPER, line_width=0.55, dash=(2.4, 1.7))
        canvas.line([(priced_x + bar_w + 4, original_top), (priced_x + bar_w + 4, adjusted_top)], stroke=COPPER, line_width=0.55)
        canvas.text(priced_x + bar_w + 8, (original_top + adjusted_top) / 2 - 2.5, f"-{liability(data, index):.2f}", size=6.6, color=COPPER)


def draw_order_track(
    canvas: Canvas,
    data: FigureData,
    highlighted: tuple[int, int],
    labels: dict[int, str],
    x: float,
    y: float,
    width: float,
) -> None:
    base_position = {index: pos for pos, index in enumerate(data.value_loss_order)}
    priced_position = {index: pos for pos, index in enumerate(data.priced_order)}
    n_accounts = len(data.accounts)
    label_w = 57
    track_x = x + label_w
    track_w = width - label_w - 6
    rows = (
        ("base order", data.value_loss_order, data.value_loss_blocked, base_position, y + 36, "cut"),
        ("priced order", data.priced_order, data.priced_blocked, priced_position, y + 8, "new cut"),
    )
    coordinates: dict[tuple[int, str], tuple[float, float]] = {}
    for row_name, order, blocked, positions, row_y, cut_label in rows:
        canvas.text(x, row_y - 3, row_name, size=7.1, color=INK)
        cut = suffix_cut(order, blocked)
        cut_x = track_x + cut * track_w / n_accounts
        if cut < n_accounts:
            canvas.rect(cut_x, row_y - 4.5, track_x + track_w - cut_x, 9, fill=SAND)
        canvas.line([(track_x, row_y), (track_x + track_w, row_y)], stroke=GRID, line_width=1.0)
        canvas.line([(cut_x, row_y - 8), (cut_x, row_y + 8)], stroke=INK, line_width=0.75)
        canvas.text(cut_x + 3, row_y + 7, cut_label, size=6.5, color=INK)
        for index in highlighted:
            marker_x = track_x + (positions[index] + 0.5) * track_w / n_accounts
            coordinates[(index, row_name)] = (marker_x, row_y)
    for index in highlighted:
        start = coordinates[(index, "base order")]
        end = coordinates[(index, "priced order")]
        canvas.line([start, end], stroke=INK, line_width=0.55, dash=None if labels[index] == "1" else (2.8, 1.8))
    for row_name, _, _, _, row_y, _ in rows:
        for index in highlighted:
            marker_x, _ = coordinates[(index, row_name)]
            draw_small_badge(canvas, marker_x, row_y + 1.2, labels[index])
    canvas.text(track_x, y - 9, "higher score", size=6.7, color=MUTED)
    canvas.text(track_x + track_w, y - 9, "lower score", size=6.7, color=MUTED, align="right")


def draw_figure_two_legend(canvas: Canvas) -> None:
    x = 12
    y = 9
    canvas.rect(x, y, 311, 27, fill=WHITE, stroke=GRID, line_width=0.6)
    canvas.rect(x + 8, y + 17, 10, 5, fill=VALUE_GREY)
    canvas.text(x + 24, y + 16.5, "value", size=7, color=INK)
    canvas.rect(x + 72, y + 17, 10, 5, fill=LOSS_RED)
    canvas.text(x + 88, y + 16.5, "loss (priced on right)", size=7, color=INK)
    canvas.line([(x + 8, y + 8), (x + 28, y + 8)], stroke=COPPER, line_width=0.55, dash=(2.4, 1.7))
    canvas.text(x + 36, y + 5, "original loss on priced bars", size=7, color=INK)


def figure_two(data: FigureData, path: Path) -> None:
    """Draw the penalty pane; Figure 1 supplies the marked baseline ranking."""
    canvas = Canvas(335, 245)
    selected_passed, selected_blocked = same_loss_example_pair(data)
    highlighted = (selected_passed, selected_blocked)
    highlight_labels = example_labels(data)

    canvas.text(167.5, 227, "Objection penalties and account order", size=10, color=INK, font="F2", align="center")
    comparison_scale = 69 / max(data.accounts[index].value + data.accounts[index].loss for index in highlighted)
    draw_penalty_comparison(
        canvas,
        data,
        selected_passed,
        13,
        129,
        comparison_scale,
        label=highlight_labels[selected_passed],
        status="base pass -> priced pass",
    )
    draw_penalty_comparison(
        canvas,
        data,
        selected_blocked,
        163,
        129,
        comparison_scale,
        label=highlight_labels[selected_blocked],
        status="base block -> priced pass",
    )
    draw_order_track(canvas, data, highlighted, highlight_labels, 15, 55, 300)
    draw_figure_two_legend(canvas)
    pdf = PDF()
    pdf.add_page(canvas)
    pdf.write(path)


def draw_grouped_account_panel(
    canvas: Canvas,
    data: FigureData,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    panel_title(canvas, x, y + height + 32, "A", "Same accounts, grouped by group")
    canvas.text(
        x,
        y + height + 15,
        "within each group: high value/loss to low; shading is the same global cut",
        size=7.2,
        color=MUTED,
    )
    group_gap = 16.0
    total_count = len(data.accounts)
    usable_width = width - group_gap
    current_x = x
    for frac in (0.25, 0.50, 0.75, 1.00):
        gy = y + height * frac
        canvas.line([(x, gy), (x + width, gy)], stroke=GRID, line_width=0.45)
    for group in (0, 1):
        order = tuple(
            sorted(
                [index for index, account in enumerate(data.accounts) if account.group == group],
                key=lambda index: value_loss_score(data.accounts[index]),
                reverse=True,
            )
        )
        group_width = usable_width * len(order) / total_count
        blocked = frozenset(index for index in order if index in data.value_loss_blocked)
        color = INDIGO if group == 0 else RUST
        canvas.text(current_x, y + height + 2, f"Group {group}", size=8.4, color=color, font="F2")
        canvas.text(
            current_x + group_width,
            y + height + 2,
            f"{len(blocked)}/{len(order)} blocked",
            size=7.2,
            color=INK,
            align="right",
        )
        draw_blocked_cut(
            canvas,
            order,
            blocked,
            current_x,
            y,
            group_width,
            height,
            label=None,
            side_labels=False,
            fill=SAND,
            stroke=INK,
            dash=None,
        )
        draw_account_bars(
            canvas,
            data,
            order,
            current_x,
            y,
            group_width,
            height * 0.80,
            blocked,
            adjusted=False,
            mark_objections=False,
            value_pass=VALUE_GREY,
            value_block=VALUE_GREY,
            loss_pass=LOSS_RED,
            loss_block=LOSS_RED,
            show_groups=True,
        )
        if group == 0:
            split_x = current_x + group_width + group_gap / 2
            canvas.line([(split_x, y), (split_x, y + height)], stroke=GRID, line_width=0.65)
        current_x += group_width + group_gap
    canvas.line([(x, y), (x + width, y), (x + width, y + height), (x, y + height), (x, y)], stroke=INK, line_width=0.65)
    canvas.rotated_text(x - 21, y + height * 0.40, "value + loss", angle_degrees=90, size=7.6, color=INK, align="center")


def draw_group_aggregation_panel(
    canvas: Canvas,
    data: FigureData,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    panel_title(canvas, x, y + height + 32, "B", "Aggregate panel A into group denominators")
    canvas.text(
        x,
        y + height + 15,
        "D_g^A = blocked objection value / all group value",
        size=7.2,
        color=MUTED,
    )
    aggregates: dict[int, tuple[float, float, float]] = {}
    for group in (0, 1):
        members = [index for index, account in enumerate(data.accounts) if account.group == group]
        total_value = sum(data.accounts[index].value for index in members)
        object_value = sum(
            data.accounts[index].value
            for index in members
            if would_object_if_blocked(data, index)
        )
        blocked_object_value = sum(
            data.accounts[index].value
            for index in members
            if index in data.value_loss_blocked and would_object_if_blocked(data, index)
        )
        aggregates[group] = (total_value, object_value, blocked_object_value)

    max_total = max(total_value for total_value, _, _ in aggregates.values())
    for row, group in enumerate((0, 1)):
        total_value, object_value, blocked_object_value = aggregates[group]
        group_y = y + height - 58 - row * 69
        bar_w = width * total_value / max_total
        bar_h = 27
        color = MUTED if group == 0 else GROUP_OUTLINE
        object_w = bar_w * object_value / total_value
        blocked_w = bar_w * blocked_object_value / total_value
        d_value = blocked_object_value / total_value

        canvas.text(x, group_y + 39, f"Group {group}", size=8.2, color=color, font="F2")
        canvas.text(x + 63, group_y + 39, "all group value", size=7.0, color=MUTED)
        canvas.rect(x, group_y, bar_w, bar_h, fill=VALUE_GREY)
        canvas.rect(x, group_y, object_w, bar_h, fill=SAND)
        canvas.rect(x, group_y, blocked_w, bar_h, fill=LOSS_RED)
        if group_has_outline(group):
            draw_group_outline(canvas, x, group_y, bar_w, bar_h, pad=0.0, include_bottom=False)
        canvas.text(x + object_w / 2, group_y + 10, "would object", size=6.6, color=COPPER, align="center")
        canvas.text(x + blocked_w + 3, group_y + 24, "blocked numerator", size=6.6, color=LOSS_RED)
        canvas.text(
            x + width,
            group_y + 11,
            f"D={d_value:.3f}",
            size=7.4,
            color=INK,
            align="right",
        )

    legend_y = y - 39
    canvas.rect(x, legend_y, width, 28, fill=WHITE, stroke=GRID, line_width=0.6)
    canvas.rect(x + 8, legend_y + 17, 10, 5, fill=VALUE_GREY)
    canvas.text(x + 23, legend_y + 16.5, "total group value", size=7.0, color=INK)
    canvas.rect(x + 119, legend_y + 17, 10, 5, fill=SAND, stroke=COPPER, line_width=0.55)
    canvas.text(x + 134, legend_y + 16.5, "would object if blocked", size=7.0, color=INK)
    canvas.rect(x + 8, legend_y + 7, 10, 5, fill=LOSS_RED)
    canvas.text(x + 23, legend_y + 6.5, "blocked objection value", size=7.0, color=INK)


def figure_three(data: FigureData, path: Path) -> None:
    canvas = Canvas(750, 305)
    canvas.text(
        375,
        281,
        "Group complaint disparity is an aggregation of account-level objections",
        size=13.0,
        color=INK,
        font="F2",
        align="center",
    )
    draw_grouped_account_panel(canvas, data, 43, 72, 342, 150)
    canvas.line([(407, 35), (407, 260)], stroke=GRID, line_width=0.8)
    draw_group_aggregation_panel(canvas, data, 431, 72, 272, 150)
    pdf = PDF()
    pdf.add_page(canvas)
    pdf.write(path)


def write_accounts_csv(data: FigureData, path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "key",
                "value",
                "loss",
                "group",
                "value_loss_score",
                "base_blocked",
                "harm_blocked",
                "net_blocked",
                "harm_objection",
                "net_objection",
                "liability",
                "priced_loss",
                "priced_value_loss_score",
                "priced_blocked",
                "figure_example_label",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        labels = example_labels(data)
        for index, account in enumerate(data.accounts):
            harm_pass, net_pass = objection_flags(data, index)
            writer.writerow(
                {
                    "key": account.key,
                    "value": f"{account.value:.6f}",
                    "loss": f"{account.loss:.6f}",
                    "group": account.group,
                    "value_loss_score": f"{value_loss_score(account):.6f}",
                    "base_blocked": int(index in data.value_loss_blocked),
                    "harm_blocked": int(index in data.harm_blocked),
                    "net_blocked": int(index in data.net_blocked),
                    "harm_objection": int(harm_pass),
                    "net_objection": int(net_pass),
                    "liability": f"{liability(data, index):.6f}",
                    "priced_loss": f"{priced_loss(data, index):.6f}",
                    "priced_value_loss_score": f"{priced_score(data, index):.6f}",
                    "priced_blocked": int(index in data.priced_blocked),
                    "figure_example_label": labels.get(index, ""),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--series", choices=("wsdm", "kdd"), default="wsdm")
    parser.add_argument(
        "--figures", nargs="+", choices=("overview", "pricing", "group-denominator"),
        default=("overview", "pricing", "group-denominator"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = build_data()
    generators = {
        "overview": (figure_one, "greedy_fairness_overview"),
        "pricing": (figure_two, "objection_pricing"),
        "group-denominator": (figure_three, "group_denominator_choice"),
    }
    for name in args.figures:
        generator, stem = generators[name]
        generator(data, args.output_dir / f"{args.series}_{stem}.pdf")
    write_accounts_csv(data, args.output_dir / "wsdm_objection_figure_accounts.csv")


if __name__ == "__main__":
    main()
