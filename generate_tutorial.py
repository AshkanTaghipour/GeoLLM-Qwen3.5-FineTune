#!/usr/bin/env python3
"""
Generate a tutorial PDF explaining the fine-tuning pipeline for Qwen 3.5-0.8B with LoRA.
Produces: ./tutorial_finetune_qwen.pdf (max 10 pages)
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.lib import colors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_FILE = "./tutorial_finetune_qwen.pdf"
HEADER_COLOR = HexColor("#2C3E50")
ACCENT_COLOR = HexColor("#2980B9")
CODE_BG = HexColor("#F4F6F7")
PAGE_W, PAGE_H = letter

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def build_styles():
    """Return a dictionary of ParagraphStyles used throughout the PDF."""
    ss = getSampleStyleSheet()

    styles = {}

    styles["Title"] = ParagraphStyle(
        "TitleCustom",
        parent=ss["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=28,
        textColor=HEADER_COLOR,
        alignment=TA_CENTER,
        spaceAfter=6,
    )

    styles["Subtitle"] = ParagraphStyle(
        "SubtitleCustom",
        parent=ss["Normal"],
        fontName="Helvetica",
        fontSize=14,
        leading=18,
        textColor=ACCENT_COLOR,
        alignment=TA_CENTER,
        spaceAfter=20,
    )

    styles["H1"] = ParagraphStyle(
        "H1Custom",
        parent=ss["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=HEADER_COLOR,
        spaceBefore=10,
        spaceAfter=8,
    )

    styles["H2"] = ParagraphStyle(
        "H2Custom",
        parent=ss["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=HEADER_COLOR,
        spaceBefore=8,
        spaceAfter=4,
    )

    styles["Body"] = ParagraphStyle(
        "BodyCustom",
        parent=ss["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=black,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )

    styles["Bullet"] = ParagraphStyle(
        "BulletCustom",
        parent=ss["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=black,
        leftIndent=20,
        bulletIndent=10,
        spaceAfter=3,
        alignment=TA_LEFT,
    )

    styles["Code"] = ParagraphStyle(
        "CodeCustom",
        parent=ss["Code"],
        fontName="Courier",
        fontSize=8,
        leading=10,
        textColor=black,
        backColor=CODE_BG,
        leftIndent=12,
        rightIndent=12,
        spaceBefore=4,
        spaceAfter=6,
        borderWidth=0.5,
        borderColor=HexColor("#D5DBDB"),
        borderPadding=6,
    )

    styles["PageNum"] = ParagraphStyle(
        "PageNum",
        parent=ss["Normal"],
        fontName="Helvetica",
        fontSize=8,
        textColor=HexColor("#7F8C8D"),
        alignment=TA_CENTER,
    )

    return styles


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def hr():
    """Horizontal rule."""
    return HRFlowable(
        width="100%", thickness=0.5, color=HexColor("#BDC3C7"),
        spaceBefore=6, spaceAfter=6,
    )


def bullet(text, sty):
    """Return a Paragraph formatted as a bullet point."""
    return Paragraph(f"\u2022  {text}", sty)


def code_block(text, sty):
    """Return a code-styled Paragraph. Newlines are converted to <br/>."""
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
        .replace("  ", "&nbsp;&nbsp;")
    )
    return Paragraph(safe, sty)


def small_spacer(h=6):
    return Spacer(1, h)


# ---------------------------------------------------------------------------
# Page-number callback
# ---------------------------------------------------------------------------

def _page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(HexColor("#7F8C8D"))
    canvas.drawCentredString(PAGE_W / 2, 30, f"Page {doc.page}")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Content builders  (one function per page)
# ---------------------------------------------------------------------------

def page1(S):
    """Title & Overview."""
    story = []
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph(
        "Fine-Tuning Qwen 3.5-0.8B with LoRA:<br/>A Practical Guide", S["Title"]
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Domain-Specific LLM Training for Mineral Exploration", S["Subtitle"]
    ))
    story.append(hr())
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "This tutorial walks through an end-to-end pipeline for fine-tuning a small "
        "large-language model (Qwen 3.5-0.8B) on domain-specific data related to "
        "mineral exploration. We use <b>LoRA</b> (Low-Rank Adaptation), a "
        "parameter-efficient method that adds tiny trainable adapters to a frozen "
        "base model, making it feasible to train on just ~150 examples without "
        "catastrophic overfitting.", S["Body"]
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>Why this approach?</b>", S["Body"]))
    story.append(bullet(
        "Qwen 3.5-0.8B is small enough to fine-tune on a single GPU yet capable "
        "enough to produce useful answers.", S["Bullet"]
    ))
    story.append(bullet(
        "LoRA keeps the vast majority of model weights frozen, adding only 1-5M "
        "trainable parameters out of 800M total.", S["Bullet"]
    ))
    story.append(bullet(
        "Domain data (mineral exploration Q&A) is scarce; parameter-efficient "
        "methods prevent memorization and preserve general knowledge.", S["Bullet"]
    ))
    story.append(bullet(
        "The pipeline covers data preparation, training, evaluation, and deployment.",
        S["Bullet"]
    ))
    story.append(PageBreak())
    return story


def page2(S):
    """Understanding the Data Pipeline."""
    story = []
    story.append(Paragraph("Understanding the Data Pipeline", S["H1"]))
    story.append(hr())

    story.append(Paragraph("<b>Two Data Types</b>", S["H2"]))
    story.append(Paragraph(
        "The training data consists of two complementary formats:", S["Body"]
    ))
    story.append(bullet(
        "<b>QA (Direct Answer)</b> -- The model receives a question and produces a "
        "concise, factual answer with no visible reasoning.", S["Bullet"]
    ))
    story.append(bullet(
        "<b>CoT (Chain-of-Thought)</b> -- The model first reasons step-by-step "
        "inside <font face='Courier'>&lt;think&gt;...&lt;/think&gt;</font> tags, "
        "then gives the final answer.", S["Bullet"]
    ))

    story.append(small_spacer(8))
    story.append(Paragraph("<b>Raw vs Transformed Format</b>", S["H2"]))
    story.append(Paragraph(
        "Raw data is stored as JSON objects with fields like <font face='Courier'>"
        "question</font>, <font face='Courier'>answer</font>, and optionally "
        "<font face='Courier'>reasoning</font>. The transformation step converts "
        "each example into the <b>chat-message format</b> expected by the model: "
        "a list of <font face='Courier'>{role, content}</font> dictionaries.",
        S["Body"]
    ))

    story.append(small_spacer(4))
    story.append(Paragraph("<b>QA example (transformed):</b>", S["Body"]))
    story.append(code_block(
        '[\n'
        '  {"role": "system", "content": "You are a mineral exploration expert."},\n'
        '  {"role": "user",   "content": "What is pyrite?"},\n'
        '  {"role": "assistant", "content": "Pyrite is an iron sulfide mineral..."}\n'
        ']', S["Code"]
    ))

    story.append(small_spacer(4))
    story.append(Paragraph("<b>CoT example (transformed):</b>", S["Body"]))
    story.append(code_block(
        '[\n'
        '  {"role": "system", "content": "You are a mineral exploration expert."},\n'
        '  {"role": "user",   "content": "Explain pyrite as a pathfinder mineral."},\n'
        '  {"role": "assistant", "content":\n'
        '      "<think>Pyrite often co-occurs with gold deposits because...\\n"\n'
        '      "So pyrite can serve as...</think>\\n"\n'
        '      "Pyrite is a key pathfinder mineral for gold exploration..."}\n'
        ']', S["Code"]
    ))

    story.append(small_spacer(4))
    story.append(Paragraph(
        "<b>Why <font face='Courier'>&lt;think&gt;</font> tags?</b>  Qwen 3.5 has "
        "a native thinking mode that uses these tags to separate internal reasoning "
        "from the final user-facing answer. By training with these tags, the model "
        "learns to produce structured reasoning that can be shown or hidden at "
        "inference time.", S["Body"]
    ))
    story.append(PageBreak())
    return story


def page3(S):
    """What is LoRA and Why Use It."""
    story = []
    story.append(Paragraph("What is LoRA and Why Use It", S["H1"]))
    story.append(hr())

    story.append(Paragraph(
        "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique. "
        "Instead of updating all 800 million parameters in Qwen 3.5-0.8B, LoRA "
        "freezes the original weights and injects small trainable matrices "
        "(called <b>adapters</b>) into specific layers of the transformer.",
        S["Body"]
    ))

    story.append(small_spacer(6))
    story.append(Paragraph("<b>Key Parameters</b>", S["H2"]))

    data = [
        ["Parameter", "Meaning", "Typical Value"],
        ["r (rank)", "Controls the size of the low-rank matrices. Higher = more\n"
         "capacity but more parameters.", "8 - 64"],
        ["alpha", "Scaling factor applied to LoRA updates.\n"
         "Effective scale = alpha / r.", "16 - 128"],
        ["Target modules", "Which linear layers receive adapters\n"
         "(e.g., q_proj, v_proj, k_proj, o_proj).", "Attention layers"],
        ["Dropout", "Dropout applied to LoRA layers to\nreduce overfitting.", "0.05 - 0.1"],
    ]
    t = Table(data, colWidths=[1.3 * inch, 3.0 * inch, 1.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEADING", (0, 0), (-1, -1), 12),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDC3C7")),
        ("BACKGROUND", (0, 1), (-1, -1), HexColor("#FAFAFA")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    story.append(small_spacer(8))
    story.append(Paragraph("<b>Why LoRA for Small Datasets?</b>", S["H2"]))
    story.append(Paragraph(
        "With only ~148 training examples, full fine-tuning would update all 800M "
        "parameters -- giving the model far too many degrees of freedom relative to "
        "the data. It would simply memorize the examples and fail to generalize. "
        "LoRA constrains the update to a low-rank subspace (typically 1-5M trainable "
        "parameters), acting as a strong implicit regularizer.",
        S["Body"]
    ))

    story.append(small_spacer(6))
    story.append(Paragraph("<b>Comparison</b>", S["H2"]))
    comp = [
        ["", "Full Fine-Tune", "LoRA"],
        ["Trainable params", "~800M (100%)", "~1-5M (<1%)"],
        ["GPU memory", "High (16+ GB)", "Low (8-12 GB)"],
        ["Overfitting risk", "Very high with small data", "Low"],
        ["Training speed", "Slower", "Faster"],
        ["Saved artifact", "Full model copy (~1.6 GB)", "Adapter files (~10-50 MB)"],
    ]
    tc = Table(comp, colWidths=[1.8 * inch, 2.0 * inch, 2.0 * inch])
    tc.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEADING", (0, 0), (-1, -1), 12),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDC3C7")),
        ("BACKGROUND", (0, 1), (-1, -1), HexColor("#FAFAFA")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(tc)
    story.append(PageBreak())
    return story


def page4(S):
    """Training Configuration Explained."""
    story = []
    story.append(Paragraph("Training Configuration Explained", S["H1"]))
    story.append(hr())

    story.append(Paragraph("<b>Batch Size &amp; Gradient Accumulation</b>", S["H2"]))
    story.append(Paragraph(
        "We cannot always fit a large batch into GPU memory, so we simulate it:",
        S["Body"]
    ))
    story.append(code_block(
        "effective_batch_size = per_device_batch_size x gradient_accumulation_steps\n"
        "                    = 4 x 4 = 16", S["Code"]
    ))
    story.append(Paragraph(
        "The model processes 4 examples at a time but accumulates gradients over 4 "
        "such micro-batches before performing a weight update. This gives the "
        "stability of a batch size of 16 with the memory cost of 4.",
        S["Body"]
    ))

    story.append(small_spacer(4))
    story.append(Paragraph("<b>Learning Rate &amp; Schedule</b>", S["H2"]))
    story.append(bullet(
        "<b>Base LR = 2e-4</b> -- This is the standard learning rate for LoRA "
        "fine-tuning. High enough to learn, low enough to avoid catastrophic "
        "forgetting.", S["Bullet"]
    ))
    story.append(bullet(
        "<b>Cosine schedule</b> -- After warmup, the learning rate follows a cosine "
        "curve, smoothly decaying to near-zero by the end of training. This helps "
        "the model settle into a stable minimum.", S["Bullet"]
    ))

    story.append(small_spacer(4))
    story.append(Paragraph("<b>Warmup</b>", S["H2"]))
    story.append(Paragraph(
        "The first 10% of training steps use a linearly increasing learning rate, "
        "starting from zero up to 2e-4. This prevents the randomly-initialized LoRA "
        "adapters from making large, destabilizing updates before the optimizer has "
        "built up momentum estimates.", S["Body"]
    ))

    story.append(small_spacer(4))
    story.append(Paragraph("<b>bf16 (bfloat16 Mixed Precision)</b>", S["H2"]))
    story.append(Paragraph(
        "We use bfloat16 rather than float32 for most computations. bfloat16 uses "
        "16 bits but allocates 8 bits to the exponent (same as float32), giving it "
        "a much wider dynamic range than fp16. This prevents the overflow/underflow "
        "issues that plague fp16 training while cutting memory usage roughly in half.",
        S["Body"]
    ))

    story.append(small_spacer(4))
    story.append(Paragraph("<b>Why Not QLoRA for Qwen 3.5?</b>", S["H2"]))
    story.append(Paragraph(
        "QLoRA quantizes the base model to 4-bit before applying LoRA adapters. "
        "While this saves additional memory, the Qwen 3.5 documentation specifically "
        "warns against it: the 0.8B model is already small, and 4-bit quantization "
        "introduces higher quantization errors that degrade performance. Standard "
        "LoRA with bf16 is the recommended approach.", S["Body"]
    ))
    story.append(PageBreak())
    return story


def page5(S):
    """Key Metrics to Monitor."""
    story = []
    story.append(Paragraph("Key Metrics to Monitor", S["H1"]))
    story.append(hr())

    story.append(Paragraph("<b>Training Loss</b>", S["H2"]))
    story.append(Paragraph(
        "The primary optimization objective. It should decrease steadily over "
        "training steps. If it plateaus early, the learning rate may be too low. "
        "If it spikes or oscillates wildly, the learning rate is too high or there "
        "is a data quality issue.", S["Body"]
    ))

    story.append(Paragraph("<b>Validation Loss</b>", S["H2"]))
    story.append(Paragraph(
        "Measured on held-out data not used for training. It should decrease and "
        "then flatten. If validation loss begins to <b>increase</b> while training "
        "loss continues to decrease, the model is <b>overfitting</b>. Remedies: "
        "reduce epochs, increase LoRA dropout, or add more training data.",
        S["Body"]
    ))

    story.append(Paragraph("<b>Gradient L2 Norm</b>", S["H2"]))
    story.append(Paragraph(
        "Measures the magnitude of parameter updates at each step. A relatively "
        "smooth gradient norm indicates stable training. Sudden spikes suggest "
        "problematic examples or numerical instability. Gradient clipping "
        "(max_grad_norm) caps these spikes.", S["Body"]
    ))

    story.append(Paragraph("<b>Learning Rate</b>", S["H2"]))
    story.append(Paragraph(
        "Visualize the scheduled learning rate to verify the warmup phase ramps up "
        "linearly and the cosine decay brings it smoothly to near-zero.",
        S["Body"]
    ))

    story.append(small_spacer(6))
    story.append(Paragraph("<b>Good vs Bad Loss Curves</b>", S["H2"]))

    curve_data = [
        ["Scenario", "Train Loss", "Val Loss", "Diagnosis"],
        ["Healthy", "Smooth decrease", "Decrease then flat", "Model is learning well"],
        ["Overfitting", "Keeps decreasing", "Decreases then rises", "Too many epochs / too few data"],
        ["Underfitting", "High, barely moves", "High, barely moves", "LR too low / model too small"],
        ["Unstable", "Spikes & oscillates", "Erratic", "LR too high / bad data batch"],
    ]
    ct = Table(curve_data, colWidths=[1.1 * inch, 1.4 * inch, 1.5 * inch, 1.8 * inch])
    ct.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEADING", (0, 0), (-1, -1), 12),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDC3C7")),
        ("BACKGROUND", (0, 1), (-1, -1), HexColor("#FAFAFA")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(ct)
    story.append(PageBreak())
    return story


def page6(S):
    """Data Transformation Details."""
    story = []
    story.append(Paragraph("Data Transformation Details", S["H1"]))
    story.append(hr())

    story.append(Paragraph("<b>Step-by-Step Walkthrough of prepare_data.py</b>", S["H2"]))
    story.append(Paragraph(
        "The data preparation script converts raw JSON files into the tokenizer-ready "
        "chat format. Here is the sequence of operations:", S["Body"]
    ))
    story.append(bullet(
        "<b>1. Load raw data</b> -- Read QA and CoT JSON files from the input directory.",
        S["Bullet"]
    ))
    story.append(bullet(
        "<b>2. Classify each example</b> -- Determine whether it is QA (no reasoning) "
        "or CoT (has a reasoning field).", S["Bullet"]
    ))
    story.append(bullet(
        "<b>3. Build chat messages</b> -- For each example, create a list of message "
        "dictionaries: system, user, and assistant.", S["Bullet"]
    ))
    story.append(bullet(
        "<b>4. Wrap CoT reasoning</b> -- For chain-of-thought examples, prepend the "
        "reasoning inside <font face='Courier'>&lt;think&gt;...&lt;/think&gt;</font> "
        "tags before the final answer.", S["Bullet"]
    ))
    story.append(bullet(
        "<b>5. Shuffle the dataset</b> -- Randomize order to prevent order-dependent "
        "pattern learning.", S["Bullet"]
    ))
    story.append(bullet(
        "<b>6. Split into train/validation</b> -- Hold out a portion for monitoring "
        "overfitting during training.", S["Bullet"]
    ))
    story.append(bullet(
        "<b>7. Save as JSONL</b> -- Write each example as one JSON object per line.",
        S["Bullet"]
    ))

    story.append(small_spacer(6))
    story.append(Paragraph("<b>How Chat Templates Work</b>", S["H2"]))
    story.append(Paragraph(
        "The tokenizer's <font face='Courier'>apply_chat_template()</font> method "
        "converts the list of messages into a specific token sequence that the model "
        "was pretrained on. This includes special tokens like "
        "<font face='Courier'>&lt;|im_start|&gt;</font> and "
        "<font face='Courier'>&lt;|im_end|&gt;</font> that delimit each role's "
        "contribution. Using the correct template is critical -- a mismatched format "
        "means the model sees gibberish.", S["Body"]
    ))

    story.append(small_spacer(4))
    story.append(Paragraph("<b>Why System Prompts Matter</b>", S["H2"]))
    story.append(Paragraph(
        "The system message sets the model's persona and behavioral expectations. "
        "For our use case, it establishes the model as a mineral exploration expert. "
        "During training the model learns to condition its outputs on this prompt, "
        "so it should match what will be used at inference time.", S["Body"]
    ))

    story.append(small_spacer(4))
    story.append(Paragraph("<b>Why We Shuffle</b>", S["H2"]))
    story.append(Paragraph(
        "If all QA examples came first and all CoT examples second, the model might "
        "learn to associate early training with one style and late training with "
        "another. Shuffling ensures each batch contains a representative mix of both "
        "types, leading to more robust learning.", S["Body"]
    ))
    story.append(PageBreak())
    return story


def page7(S):
    """The Training Loop."""
    story = []
    story.append(Paragraph("The Training Loop (Under the Hood)", S["H1"]))
    story.append(hr())

    story.append(Paragraph(
        "Each training step performs the following sequence:", S["Body"]
    ))

    story.append(small_spacer(4))
    story.append(Paragraph("<b>1. Forward Pass</b>", S["H2"]))
    story.append(Paragraph(
        "Input tokens are fed through the transformer. Each layer computes attention "
        "and feed-forward operations. The LoRA adapters add their low-rank "
        "contributions to the frozen weight matrices. The output is a probability "
        "distribution over the vocabulary for each position.", S["Body"]
    ))

    story.append(Paragraph("<b>2. Loss Computation</b>", S["H2"]))
    story.append(Paragraph(
        "Cross-entropy loss is computed between the predicted next-token probabilities "
        "and the actual next tokens. Crucially, the loss is computed <b>only on "
        "assistant response tokens</b>, not on user or system tokens. This teaches "
        "the model to generate good answers without trying to learn how to generate "
        "questions.", S["Body"]
    ))

    story.append(Paragraph("<b>3. Backward Pass</b>", S["H2"]))
    story.append(Paragraph(
        "Gradients are computed via backpropagation. Because the base model weights "
        "are frozen, gradients only flow through and update the LoRA adapter matrices. "
        "This dramatically reduces the computation and memory required.", S["Body"]
    ))

    story.append(Paragraph("<b>4. Optimizer Step (AdamW-8bit)</b>", S["H2"]))
    story.append(Paragraph(
        "The AdamW optimizer updates LoRA weights using first and second moment "
        "estimates. The 8-bit variant stores optimizer states in 8-bit precision, "
        "cutting optimizer memory by ~75% with negligible impact on convergence. "
        "Weight decay (the 'W' in AdamW) applies L2 regularization to prevent "
        "weights from growing too large.", S["Body"]
    ))

    story.append(Paragraph("<b>5. Gradient Checkpointing</b>", S["H2"]))
    story.append(Paragraph(
        "Normally, all intermediate activations are stored during the forward pass "
        "so they can be used during backpropagation. Gradient checkpointing discards "
        "most of these activations and <b>recomputes</b> them during the backward "
        "pass. This trades ~30% more compute time for ~60% less memory, making it "
        "possible to train on longer sequences or with larger batch sizes.",
        S["Body"]
    ))

    story.append(small_spacer(6))
    story.append(Paragraph("<b>Putting It Together</b>", S["H2"]))
    story.append(code_block(
        "for each epoch:\n"
        "    for each batch of 4 examples:\n"
        "        logits = model(input_ids)           # forward pass\n"
        "        loss = cross_entropy(logits, labels) # loss on assistant tokens only\n"
        "        loss.backward()                      # compute gradients (LoRA only)\n"
        "        if step % grad_accum == 0:           # every 4 micro-batches\n"
        "            optimizer.step()                 # update LoRA weights\n"
        "            scheduler.step()                 # adjust learning rate\n"
        "            optimizer.zero_grad()            # reset gradients",
        S["Code"]
    ))
    story.append(PageBreak())
    return story


def page8(S):
    """After Training -- Saving and Using the Model."""
    story = []
    story.append(Paragraph("After Training: Saving and Using the Model", S["H1"]))
    story.append(hr())

    story.append(Paragraph("<b>LoRA Adapter Files</b>", S["H2"]))
    story.append(Paragraph(
        "After training completes, the LoRA adapters are saved as small files "
        "(typically 10-50 MB). These contain only the trained low-rank matrices, not "
        "the full model. To use them, you load the base Qwen model and apply the "
        "adapter on top.", S["Body"]
    ))
    story.append(code_block(
        "from peft import PeftModel\n"
        "from transformers import AutoModelForCausalLM\n\n"
        "base_model = AutoModelForCausalLM.from_pretrained(\"Qwen/Qwen3.5-0.8B\")\n"
        "model = PeftModel.from_pretrained(base_model, \"./lora_adapters\")",
        S["Code"]
    ))

    story.append(small_spacer(4))
    story.append(Paragraph("<b>Merged Model</b>", S["H2"]))
    story.append(Paragraph(
        "For simpler deployment, you can merge the LoRA weights back into the base "
        "model. This produces a single model that behaves identically but does not "
        "require the PEFT library at inference time.", S["Body"]
    ))
    story.append(code_block(
        "merged = model.merge_and_unload()\n"
        "merged.save_pretrained(\"./merged_model\")\n"
        "tokenizer.save_pretrained(\"./merged_model\")", S["Code"]
    ))

    story.append(small_spacer(4))
    story.append(Paragraph("<b>Inference Example</b>", S["H2"]))
    story.append(Paragraph(
        "Once the model is loaded (adapter or merged), generating predictions follows "
        "the standard Hugging Face pattern:", S["Body"]
    ))
    story.append(code_block(
        "from transformers import AutoTokenizer, AutoModelForCausalLM\n\n"
        "tokenizer = AutoTokenizer.from_pretrained(\"./merged_model\")\n"
        "model = AutoModelForCausalLM.from_pretrained(\"./merged_model\")\n\n"
        "messages = [\n"
        '    {"role": "system", "content": "You are a mineral exploration expert."},\n'
        '    {"role": "user", "content": "What minerals indicate gold proximity?"}\n'
        "]\n\n"
        "input_ids = tokenizer.apply_chat_template(\n"
        "    messages, return_tensors=\"pt\", add_generation_prompt=True\n"
        ")\n"
        "output = model.generate(input_ids, max_new_tokens=512, temperature=0.7)\n"
        "print(tokenizer.decode(output[0], skip_special_tokens=True))",
        S["Code"]
    ))

    story.append(small_spacer(6))
    story.append(Paragraph("<b>Deployment Considerations</b>", S["H2"]))
    story.append(bullet(
        "<b>Adapter approach</b>: Smaller storage, can swap adapters for different "
        "domains, requires PEFT at runtime.", S["Bullet"]
    ))
    story.append(bullet(
        "<b>Merged approach</b>: Larger storage (~1.6 GB), simpler deployment, no "
        "PEFT dependency, slightly faster inference.", S["Bullet"]
    ))
    story.append(PageBreak())
    return story


def page9(S):
    """Evaluation Strategy."""
    story = []
    story.append(Paragraph("Evaluation Strategy", S["H1"]))
    story.append(hr())

    story.append(Paragraph("<b>Why Evaluation Matters</b>", S["H2"]))
    story.append(Paragraph(
        "Fine-tuning can cause a model to memorize training data, hallucinate domain "
        "facts, or lose general-purpose capabilities. A structured evaluation ensures "
        "the model has genuinely learned useful domain knowledge while remaining "
        "reliable.", S["Body"]
    ))

    story.append(small_spacer(4))
    story.append(Paragraph("<b>The Scoring Rubric</b>", S["H2"]))
    story.append(Paragraph(
        "Each model response is scored on five dimensions:", S["Body"]
    ))

    rubric = [
        ["Criterion", "What It Measures", "Weight"],
        ["Technical Accuracy", "Are the geological/mineralogical facts correct?", "High"],
        ["Reasoning Quality", "Is the chain-of-thought logical and coherent?", "Medium"],
        ["Completeness", "Does the answer address all parts of the question?", "Medium"],
        ["Specificity", "Does it provide concrete details, not vague generalities?", "Medium"],
        ["Hallucination Control", "Does it avoid fabricating information?", "High"],
    ]
    rt = Table(rubric, colWidths=[1.4 * inch, 3.0 * inch, 1.0 * inch])
    rt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEADING", (0, 0), (-1, -1), 12),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDC3C7")),
        ("BACKGROUND", (0, 1), (-1, -1), HexColor("#FAFAFA")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(rt)

    story.append(small_spacer(6))
    story.append(Paragraph("<b>Hallucination Traps</b>", S["H2"]))
    story.append(Paragraph(
        "The evaluation includes deliberately tricky questions designed to catch the "
        "model making things up:", S["Body"]
    ))
    story.append(bullet(
        "Questions about fictional minerals or deposits that do not exist -- the "
        "model should say it does not know.", S["Bullet"]
    ))
    story.append(bullet(
        "Questions that mix real and fake information -- the model should identify "
        "and reject the false parts.", S["Bullet"]
    ))
    story.append(bullet(
        "Questions outside the training domain -- the model should acknowledge the "
        "limits of its expertise.", S["Bullet"]
    ))

    story.append(small_spacer(6))
    story.append(Paragraph("<b>Base vs Fine-Tuned Comparison</b>", S["H2"]))
    story.append(Paragraph(
        "The same set of evaluation questions is posed to both the original Qwen "
        "3.5-0.8B base model and the fine-tuned version. Responses are scored using "
        "the rubric above, and the results are compared side-by-side. This "
        "quantifies the improvement from fine-tuning and reveals any regressions "
        "in general capability.", S["Body"]
    ))
    story.append(PageBreak())
    return story


def page10(S):
    """Common Issues and Troubleshooting."""
    story = []
    story.append(Paragraph("Common Issues and Troubleshooting", S["H1"]))
    story.append(hr())

    issues = [
        (
            "Loss Not Decreasing",
            "Check that the data format matches the chat template exactly. Verify "
            "the learning rate is not too low (try 2e-4 to 5e-4). Ensure the loss is "
            "computed on the correct tokens (assistant only). Inspect a few training "
            "examples to rule out data corruption."
        ),
        (
            "Overfitting (Validation Loss Increasing)",
            "Reduce the number of training epochs (start with 3). Increase LoRA "
            "dropout (e.g., from 0.05 to 0.1). Add more training examples if "
            "possible. Reduce the LoRA rank to further constrain capacity."
        ),
        (
            "Out of Memory (OOM)",
            "Reduce per_device_train_batch_size (try 2 or 1). Reduce "
            "max_seq_length (e.g., from 2048 to 1024). Enable gradient "
            "checkpointing if not already on. Use bf16 to halve activation memory. "
            "Reduce LoRA rank."
        ),
        (
            "Model Repeating Itself",
            "This is usually an inference-time issue, not a training problem. Use "
            "repetition_penalty (e.g., 1.1-1.2) during generation. Adjust "
            "temperature (0.6-0.8) and top_p (0.9). If it persists, check that the "
            "chat template terminates correctly with an end-of-turn token."
        ),
        (
            "Model Forgot General Knowledge (Catastrophic Forgetting)",
            "Too many training epochs or too high a learning rate caused the model "
            "to overwrite its pretrained knowledge. Remedies: use fewer epochs "
            "(1-3), lower the learning rate, or reduce the LoRA rank. Consider "
            "mixing in a small amount of general-purpose data during training."
        ),
    ]

    for title, desc in issues:
        story.append(Paragraph(f"<b>{title}</b>", S["H2"]))
        story.append(Paragraph(desc, S["Body"]))
        story.append(small_spacer(4))

    story.append(hr())
    story.append(small_spacer(8))
    story.append(Paragraph(
        "<b>Quick Reference Checklist</b>", S["H2"]
    ))
    story.append(bullet("Verify data format with a manual inspection before training.", S["Bullet"]))
    story.append(bullet("Start with conservative hyperparameters (3 epochs, lr=2e-4, r=16).", S["Bullet"]))
    story.append(bullet("Monitor validation loss -- stop if it rises for 2+ evaluations.", S["Bullet"]))
    story.append(bullet("Save checkpoints frequently so you can roll back.", S["Bullet"]))
    story.append(bullet("Always compare against the base model to quantify improvement.", S["Bullet"]))
    story.append(bullet("Test with hallucination traps before deploying.", S["Bullet"]))

    return story


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    styles = build_styles()

    doc = SimpleDocTemplate(
        OUTPUT_FILE,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Fine-Tuning Qwen 3.5-0.8B with LoRA",
        author="Tutorial Generator",
    )

    story = []
    for page_fn in [page1, page2, page3, page4, page5, page6, page7, page8, page9, page10]:
        story.extend(page_fn(styles))

    doc.build(story, onFirstPage=_page_number, onLaterPages=_page_number)
    print(f"Tutorial PDF generated: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
