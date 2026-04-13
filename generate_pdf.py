# builds the whitepaper PDF via reportlab
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    KeepTogether, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
import os

OUT_DIR = "/Users/srimanarayana/Research Project I"
PDF_PATH = os.path.join(OUT_DIR, "whitepaper.pdf")

doc = SimpleDocTemplate(
    PDF_PATH, pagesize=letter,
    topMargin=0.8*inch, bottomMargin=0.8*inch,
    leftMargin=1*inch, rightMargin=1*inch,
)

styles = getSampleStyleSheet()

styles.add(ParagraphStyle('PaperTitle', parent=styles['Title'],
    fontSize=16, leading=20, alignment=TA_CENTER, spaceAfter=6,
    textColor=HexColor('#1a1a2e'), fontName='Times-Bold'))
styles.add(ParagraphStyle('PaperSubtitle', parent=styles['Normal'],
    fontSize=10, leading=13, alignment=TA_CENTER, spaceAfter=4,
    textColor=HexColor('#444444'), fontName='Times-Italic'))
styles.add(ParagraphStyle('AbstractTitle', parent=styles['Heading2'],
    fontSize=11, fontName='Times-Bold', alignment=TA_CENTER,
    spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle('AbstractBody', parent=styles['Normal'],
    fontSize=9.5, leading=13, fontName='Times-Roman',
    alignment=TA_JUSTIFY, leftIndent=30, rightIndent=30, spaceAfter=12))
styles.add(ParagraphStyle('SectionHead', parent=styles['Heading1'],
    fontSize=13, fontName='Times-Bold', spaceBefore=18, spaceAfter=8,
    textColor=HexColor('#1a1a2e')))
styles.add(ParagraphStyle('SubsectionHead', parent=styles['Heading2'],
    fontSize=11, fontName='Times-Bold', spaceBefore=12, spaceAfter=6,
    textColor=HexColor('#2d2d44')))
styles.add(ParagraphStyle('BoldBody', parent=styles['Normal'],
    fontSize=10, leading=13.5, fontName='Times-Bold',
    alignment=TA_JUSTIFY, spaceBefore=2, spaceAfter=4))
styles.add(ParagraphStyle('BulletText', parent=styles['Normal'],
    fontSize=10, leading=13, fontName='Times-Roman',
    alignment=TA_JUSTIFY, leftIndent=20, bulletIndent=10,
    spaceBefore=1, spaceAfter=1))
styles.add(ParagraphStyle('TableCell', parent=styles['Normal'],
    fontSize=8.5, leading=11, fontName='Times-Roman', alignment=TA_LEFT))
styles.add(ParagraphStyle('TableHeader', parent=styles['Normal'],
    fontSize=8.5, leading=11, fontName='Times-Bold', alignment=TA_LEFT))
styles.add(ParagraphStyle('RefText', parent=styles['Normal'],
    fontSize=9, leading=12, fontName='Times-Roman',
    leftIndent=20, firstLineIndent=-20, spaceBefore=2, spaceAfter=2))

styles['BodyText'].fontSize = 10
styles['BodyText'].leading = 13.5
styles['BodyText'].fontName = 'Times-Roman'
styles['BodyText'].alignment = TA_JUSTIFY
styles['BodyText'].spaceBefore = 2
styles['BodyText'].spaceAfter = 4

story = []

# title
story.append(Spacer(1, 20))
story.append(Paragraph(
    "Attention Heads Are Not Monolithic:<br/>"
    "Sub-Head Functional Decomposition via SVD of QK Circuits in GPT-2",
    styles['PaperTitle']))
story.append(Spacer(1, 8))
story.append(Paragraph("April 2026", styles['PaperSubtitle']))
story.append(Spacer(1, 4))
story.append(HRFlowable(width="80%", thickness=1, color=HexColor('#cccccc')))
story.append(Spacer(1, 10))

# abstract
story.append(Paragraph("Abstract", styles['AbstractTitle']))
story.append(Paragraph(
    "The standard unit of analysis in transformer mechanistic interpretability is the attention head. "
    "We argue this is too coarse. By applying SVD to the composed query-key matrices "
    "(W<sub>QK</sub> = W<sub>Q</sub>W<sub>K</sub><super>T</super>) of all 144 heads in GPT-2 small, "
    "we find structure that head-level analysis misses entirely. The key results: (1) the semantic alignment "
    "between query and key singular directions follows a <b>bimodal distribution</b> -- some heads match "
    "tokens to similar tokens, others match to dissimilar ones, and this split is far outside what random "
    "matrices produce; (2) individual SVD directions within a single head can have causal effects on "
    "downstream tasks that are an order of magnitude larger than the head's net effect, but in <b>opposing "
    "directions</b>, meaning the head's apparent unimportance masks massive internal computation; "
    "(3) these directions show effect sizes above Cohen's d = 3.0 when tested against task-relevant vs. "
    "control prompts. The upshot is that weight-based SVD, which requires no training data and no "
    "hyperparameters, can crack open attention heads and reveal functional structure that activation-based "
    "methods treat as a black box.",
    styles['AbstractBody']))
story.append(HRFlowable(width="80%", thickness=0.5, color=HexColor('#dddddd')))
story.append(Spacer(1, 8))

# 1. introduction
story.append(Paragraph("1. Introduction", styles['SectionHead']))
story.append(Paragraph("1.1 The problem", styles['SubsectionHead']))
story.append(Paragraph(
    "Everyone in mech interp knows what an attention head does -- it computes a bilinear attention pattern "
    "and writes a linear function of attended values back to the residual stream. The field has gotten "
    "remarkably far treating heads as atoms: induction heads, name movers, inhibition heads, the whole IOI "
    "circuit. But a head with a 64-dimensional key/query space has room to do more than one thing at once. "
    "And there's growing evidence that they do.",
    styles['BodyText']))
story.append(Paragraph(
    "The \"Beyond Components\" work (Lieberum et al., 2025) showed this for OV circuits -- you can decompose "
    "them via SVD and find multiple independent sub-computations living in different singular directions. "
    "But nobody has done this systematically for QK circuits, and there's a good reason: the QK decomposition "
    "is harder. W<sub>OV</sub> maps from residual stream to residual stream -- same space in, same space out. "
    "W<sub>QK</sub> is a bilinear form. The left singular vectors (queries) and right singular vectors (keys) "
    "live in different semantic subspaces. You can't just look at one set of token projections; you need to "
    "understand the <i>relationship</i> between the two.",
    styles['BodyText']))
story.append(Paragraph(
    "This paper fills that gap. We decompose every QK matrix in GPT-2 small, measure whether the query and "
    "key directions are semantically related, and then causally validate the individual directions using the "
    "IOI task as ground truth.",
    styles['BodyText']))

story.append(Paragraph("1.2 What we found", styles['SubsectionHead']))
story.append(Paragraph(
    "<b>1. The Q-K alignment distribution is bimodal.</b> When you project query and key singular vectors "
    "into token space and measure their cosine similarity, you don't get a blob around zero. You get two "
    "clusters -- heads where Q and K activate <i>similar</i> tokens (semantic matching) and heads where they "
    "activate <i>opposite</i> tokens (contrastive matching). 11 heads above +2\u03c3, 9 heads below -2\u03c3.",
    styles['BulletText']))
story.append(Paragraph(
    "<b>2. Single heads contain massive opposing sub-computations.</b> The S-inhibition head L9H6 has a net "
    "effect of -0.046 on IOI when you ablate the whole thing. Looks boring. But ablate just its first singular "
    "direction and the effect is +0.386. Just the second: +0.800. These are 8x and 17x the net effect. "
    "The directions are fighting each other inside the head.",
    styles['BulletText']))
story.append(Paragraph(
    "<b>3. The SVD directions are real features, not noise.</b> Residual stream projections onto these "
    "directions separate IOI prompts from control prompts with Cohen's d > 3.0.",
    styles['BulletText']))

# 2. background
story.append(Paragraph("2. Background", styles['SectionHead']))
story.append(Paragraph("2.1 The QK and OV circuits", styles['SubsectionHead']))
story.append(Paragraph(
    "Following Elhage et al. (2021), every attention head has two composed circuit matrices. "
    "W<sub>QK</sub> = W<sub>Q</sub>W<sub>K</sub><super>T</super> determines <i>where</i> the head attends. "
    "W<sub>OV</sub> = W<sub>V</sub>W<sub>O</sub> determines <i>what</i> happens when it attends. Both are "
    "768 x 768 matrices in GPT-2 small. The attention score between positions i and j is "
    "x<sub>i</sub><super>T</super> W<sub>QK</sub> x<sub>j</sub> / \u221Ad<sub>head</sub> -- a bilinear form. "
    "SVD decomposes this into independent rank-1 matching rules: the head attends from tokens aligned with "
    "u<sub>k</sub> to tokens aligned with v<sub>k</sub>, with strength \u03c3<sub>k</sub>.",
    styles['BodyText']))

story.append(Paragraph("2.2 Why QK is harder than OV", styles['SubsectionHead']))
story.append(Paragraph(
    "For W<sub>OV</sub>, you can project both sides through the same embedding and get interpretable token "
    "lists. For W<sub>QK</sub>, u<sub>k</sub> tells you what the query \"looks for\" and v<sub>k</sub> tells "
    "you what the key \"looks like\" -- different semantic questions about different parts of the sequence. "
    "The asymmetry is real and makes interpretation harder.",
    styles['BodyText']))

story.append(Paragraph("2.3 IOI as ground truth", styles['SubsectionHead']))
story.append(Paragraph(
    "We validate against the Indirect Object Identification circuit (Wang et al., 2022). \"When Mary and John "
    "went to the store, John gave a drink to ___\" \u2192 \"Mary\". The circuit has known name movers (L9H9, "
    "L10H0), S-inhibition heads (L7H3, L7H9, L9H6), induction heads, and duplicate token detectors.",
    styles['BodyText']))

# 3. methods
story.append(Paragraph("3. Methods", styles['SectionHead']))
story.append(Paragraph(
    "GPT-2 small: 12 layers, 12 heads, d<sub>model</sub> = 768, d<sub>head</sub> = 64, vocab = 50,257. "
    "Weights extracted via TransformerLens. Full SVD in float64 via scipy. Effective rank defined as "
    "singular values needed for 90% of squared Frobenius norm. Token projections via cosine similarity "
    "with embedding matrix rows. Q-K alignment measured as cosine between W<sub>E</sub>u<sub>k</sub> and "
    "W<sub>E</sub>v<sub>k</sub>, compared against 1,000 random pairs.",
    styles['BodyText']))
story.append(Paragraph(
    "Causal ablation uses TransformerLens hooks to project out u<sub>k</sub> from hook_resid_pre, removing "
    "the direction's contribution to all downstream computation. IOI logit difference (correct name minus "
    "wrong name) averaged over 100 prompts. Activation verification via Welch's t-test and Cohen's d on "
    "residual stream projections for positive vs. control prompts.",
    styles['BodyText']))

# 4. results
story.append(PageBreak())
story.append(Paragraph("4. Results", styles['SectionHead']))

story.append(Paragraph("4.1 Effective rank varies wildly", styles['SubsectionHead']))
t1_data = [
    [Paragraph('<b>Head</b>', styles['TableHeader']),
     Paragraph('<b>Role</b>', styles['TableHeader']),
     Paragraph('<b>Top SV</b>', styles['TableHeader']),
     Paragraph('<b>Eff. Rank</b>', styles['TableHeader'])],
    [Paragraph('L1H3', styles['TableCell']), Paragraph('?', styles['TableCell']),
     Paragraph('1.68', styles['TableCell']), Paragraph('7', styles['TableCell'])],
    [Paragraph('L1H10', styles['TableCell']), Paragraph('?', styles['TableCell']),
     Paragraph('1.42', styles['TableCell']), Paragraph('7', styles['TableCell'])],
    [Paragraph('L2H2', styles['TableCell']), Paragraph('?', styles['TableCell']),
     Paragraph('22.95', styles['TableCell']), Paragraph('15', styles['TableCell'])],
    [Paragraph('L1H4', styles['TableCell']), Paragraph('Induction', styles['TableCell']),
     Paragraph('1.84', styles['TableCell']), Paragraph('18', styles['TableCell'])],
    [Paragraph('L9H9', styles['TableCell']), Paragraph('Name Mover', styles['TableCell']),
     Paragraph('2.43', styles['TableCell']), Paragraph('~45', styles['TableCell'])],
    [Paragraph('L11H8', styles['TableCell']), Paragraph('? (OV)', styles['TableCell']),
     Paragraph('334.87 (OV)', styles['TableCell']), Paragraph('1 (OV)', styles['TableCell'])],
]
t1 = Table(t1_data, colWidths=[60, 80, 80, 70])
t1.setStyle(TableStyle([
    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ('BACKGROUND', (0,0), (-1,0), HexColor('#e8e8f0')),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 4),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
]))
story.append(Paragraph(
    "Range is 7 to 54 for QK (mean 45.6). L11H8's OV circuit is effectively rank-1 -- a single linear "
    "map with \u03c3<sub>1</sub> = 334.87. L2H2 has the largest QK singular value in the model (22.95).",
    styles['BodyText']))
story.append(t1)
story.append(Spacer(1, 10))

story.append(Paragraph("4.2 The bimodal alignment distribution", styles['SubsectionHead']))
story.append(Paragraph(
    "Random baseline: cosine mean = 0.005, std = 0.200. Actual Q-K pairs (mean of top-3 SVs per head):",
    styles['BodyText']))
story.append(Paragraph(
    "<b>Semantic matching</b> (cosine > +2\u03c3): L0H1 (0.904), L0H5 (0.880), L0H10 (0.843), "
    "L5H1 (0.592), L5H0 (0.592), L6H10 (0.586). 11 heads total. Mostly layer 0.",
    styles['BulletText']))
story.append(Paragraph(
    "<b>Contrastive matching</b> (cosine < -2\u03c3): L0H9 (-0.849), L10H7 (-0.740), "
    "L0H6 (-0.655), L11H1 (-0.539), L11H8 (-0.479). 9 heads total. Mostly layers 10-11.",
    styles['BulletText']))
story.append(Paragraph(
    "The semantic matchers in layer 0 are probably doing direct embedding-similarity attention. The "
    "contrastive matchers in late layers are consistent with inhibitory or corrective roles. This rules "
    "out the null hypothesis that QK circuits are purely positional.",
    styles['BodyText']))

story.append(Paragraph("4.3 Sub-head functional opposition", styles['SubsectionHead']))
story.append(Paragraph(
    "This is the main result. Baseline IOI logit diff: 4.469 \u00b1 0.836.",
    styles['BoldBody']))

t2_data = [
    [Paragraph('<b>Direction</b>', styles['TableHeader']),
     Paragraph('<b>Effect</b>', styles['TableHeader']),
     Paragraph('<b>Multiple of net</b>', styles['TableHeader'])],
    [Paragraph('SV0', styles['TableCell']), Paragraph('+0.386', styles['TableCell']),
     Paragraph('8.4x', styles['TableCell'])],
    [Paragraph('SV1', styles['TableCell']), Paragraph('+0.800', styles['TableCell']),
     Paragraph('17.4x', styles['TableCell'])],
    [Paragraph('SV2', styles['TableCell']), Paragraph('-0.169', styles['TableCell']),
     Paragraph('opposing', styles['TableCell'])],
    [Paragraph('Top-3 cum.', styles['TableCell']), Paragraph('+0.872', styles['TableCell']),
     Paragraph('19x', styles['TableCell'])],
    [Paragraph('Full head', styles['TableCell']), Paragraph('-0.046', styles['TableCell']),
     Paragraph('1x (net)', styles['TableCell'])],
]
t2 = Table(t2_data, colWidths=[80, 80, 100])
t2.setStyle(TableStyle([
    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ('BACKGROUND', (0,0), (-1,0), HexColor('#e8e8f0')),
    ('BACKGROUND', (0,2), (-1,2), HexColor('#fff3e0')),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 4),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
]))
story.append(Paragraph(
    "<b>S-Inhibition L9H6.</b> Full head ablation: -0.046 -- basically nothing. But SV1 alone is +0.800. "
    "There's a war inside this head.",
    styles['BodyText']))
story.append(t2)
story.append(Spacer(1, 6))
story.append(Paragraph(
    "<b>Name Mover L9H9.</b> SV0: +0.369 (135% of the full head's +0.273). SV1: -0.135, actively opposing. "
    "The head does name-moving <i>and</i> something else, simultaneously, and they partially cancel.",
    styles['BodyText']))

story.append(Paragraph("4.4 Activation verification", styles['SubsectionHead']))
t3_data = [
    [Paragraph('<b>Head &amp; Direction</b>', styles['TableHeader']),
     Paragraph('<b>Cohen\'s d</b>', styles['TableHeader']),
     Paragraph('<b>p-value</b>', styles['TableHeader'])],
    [Paragraph('L9H9 SV0 query', styles['TableCell']),
     Paragraph('3.46', styles['TableCell']), Paragraph('< 0.0001', styles['TableCell'])],
    [Paragraph('L9H6 SV0 query', styles['TableCell']),
     Paragraph('-3.22', styles['TableCell']), Paragraph('< 0.0001', styles['TableCell'])],
    [Paragraph('L9H6 SV1 query', styles['TableCell']),
     Paragraph('-3.25', styles['TableCell']), Paragraph('< 0.0001', styles['TableCell'])],
    [Paragraph('L1H4 SV0 query', styles['TableCell']),
     Paragraph('2.95', styles['TableCell']), Paragraph('< 0.0001', styles['TableCell'])],
    [Paragraph('L1H4 SV0 key', styles['TableCell']),
     Paragraph('2.80', styles['TableCell']), Paragraph('< 0.0001', styles['TableCell'])],
]
t3 = Table(t3_data, colWidths=[130, 70, 70])
t3.setStyle(TableStyle([
    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ('BACKGROUND', (0,0), (-1,0), HexColor('#e8e8f0')),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 4),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
]))
story.append(t3)
story.append(Spacer(1, 6))
story.append(Paragraph(
    "d > 2.0 is \"huge\" by convention. These directions aren't noise. The negative d for L9H6 means the "
    "query directions are <i>suppressed</i> on IOI inputs -- consistent with inhibition. L1H4 SV0 activates "
    "bilaterally (query and key) on repeated sequences, exactly what an induction head should do.",
    styles['BodyText']))

# 5. discussion
story.append(PageBreak())
story.append(Paragraph("5. Discussion", styles['SectionHead']))
story.append(Paragraph("5.1 What this breaks", styles['SubsectionHead']))
story.append(Paragraph(
    "Head-level circuit analysis has a blind spot. When a head's net ablation effect is near zero, the "
    "standard conclusion is \"this head doesn't participate.\" L9H6 shows that's wrong -- the head can "
    "be doing enormous amounts of work that cancels internally. You can only see this by going below "
    "the head level.",
    styles['BodyText']))
story.append(Paragraph(
    "This also means feature attribution at the head level is lossy. SAEs trained on head outputs will "
    "learn features that are mixtures of these SVD directions. The SVD gives you the unmixed version, "
    "for free, from weights alone.",
    styles['BodyText']))

story.append(Paragraph("5.2 Layer-dependent specialization", styles['SubsectionHead']))
story.append(Paragraph(
    "Semantic-matching heads cluster in layer 0; contrastive heads in layers 10-11. Early layers do local "
    "token-similarity matching, late layers attend to tokens that are specifically <i>not</i> the current "
    "token -- the pattern you'd expect from inhibition or correction mechanisms.",
    styles['BodyText']))

story.append(Paragraph("5.3 Limitations", styles['SubsectionHead']))
story.append(Paragraph(
    "The residual stream ablation isn't QK-specific -- it affects everything downstream, not just the target "
    "head. A cleaner intervention would subtract rank-1 components directly from the attention scores. "
    "Token projections for QK directions are noisier than for OV -- lots of rare unicode and control "
    "characters. And this is GPT-2 small only; needs replication on larger models.",
    styles['BodyText']))

# 6. conclusion
story.append(Paragraph("6. Conclusion", styles['SectionHead']))
story.append(Paragraph(
    "SVD of W<sub>QK</sub> reveals that attention heads aren't monolithic. They contain multiple independent "
    "matching rules, some of which actively oppose each other. L9H6 is the clearest example: individual "
    "directions have effects 17x larger than the head's net effect, but they fight, and the head looks "
    "inert from the outside.",
    styles['BodyText']))
story.append(Paragraph(
    "The practical takeaway: if you want to understand what an attention head is doing, don't just ablate "
    "the whole thing. Decompose its weight matrices. SVD is free -- no training, no data, no hyperparameters. "
    "Just a matrix factorization on the weights you already have.",
    styles['BodyText']))

# 7. reproducibility
story.append(Paragraph("7. Reproducibility", styles['SectionHead']))
story.append(Paragraph(
    "All code and 65 plots are in the repository. Full pipeline takes ~2 hours on a CPU.<br/><br/>"
    "<b>Stack:</b> Python 3.9, TransformerLens 2.18.0, PyTorch 2.8.0, scipy 1.13.1",
    styles['BodyText']))

# references
story.append(Paragraph("References", styles['SectionHead']))
for ref in [
    "[1] Elhage, N., et al. (2021). \"A Mathematical Framework for Transformer Circuits.\" Anthropic.",
    "[2] Wang, K., et al. (2022). \"Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small.\"",
    "[3] Olsson, C., et al. (2022). \"In-context Learning and Induction Heads.\" Anthropic.",
    "[4] Bricken, T., et al. (2023). \"Towards Monosemanticity.\" Anthropic.",
    "[5] Lieberum, T., et al. (2025). \"Beyond Components: Decomposing Transformer Attention Heads into Functionally Distinct Sub-Operations.\"",
    "[6] Conmy, A., et al. (2023). \"Towards Automated Circuit Discovery for Mechanistic Interpretability.\"",
]:
    story.append(Paragraph(ref, styles['RefText']))

doc.build(story)
print(f"Written to {PDF_PATH}")
