"""Build the separate, explicitly exploratory defense-slide draft."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "deliverables" / "SOUTENANCE_VERSION_TRAVAIL.pptx"
COLORS = {
    "navy": "13283F",
    "teal": "087E8B",
    "aqua": "DDF3F1",
    "ink": "203040",
    "muted": "526477",
    "paper": "F7F9FC",
    "white": "FFFFFF",
    "amber": "E6A23C",
    "red": "A63D40",
    "line": "D8E0E8",
    "blue": "3977A8",
    "green": "477A68",
    "purple": "8064A2",
}


def color(value: str) -> RGBColor:
    return RGBColor.from_string(COLORS.get(value, value))


def shape(slide, x, y, width, height, fill, rounded=True):
    item = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
        Inches(x),
        Inches(y),
        Inches(width),
        Inches(height),
    )
    item.fill.solid()
    item.fill.fore_color.rgb = color(fill)
    item.line.fill.background()
    return item


def add_text(
    slide,
    content,
    x,
    y,
    width,
    height,
    size=18,
    fill="ink",
    bold=False,
    margin=0.06,
):
    box = slide.shapes.add_textbox(
        Inches(x), Inches(y), Inches(width), Inches(height)
    )
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    for index, line in enumerate(content.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.font.name = "Aptos"
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = color(fill)
        paragraph.space_after = Pt(5)
    return box


def slide(title, section=""):
    current = PRESENTATION.slides.add_slide(PRESENTATION.slide_layouts[6])
    current.background.fill.solid()
    current.background.fill.fore_color.rgb = color("paper")
    shape(current, 0, 0, 13.333, 0.12, "teal", rounded=False)
    if section:
        add_text(current, section.upper(), 0.55, 0.32, 4, 0.28, 10, "teal", True)
    add_text(current, title, 0.55, 0.65, 12.2, 0.62, 28, "navy", True)
    shape(current, 0.55, 1.38, 1.05, 0.045, "amber", rounded=False)
    add_text(
        current,
        "VERSION DE TRAVAIL  •  Résultats historiques = EXPLORATOIRES  •  Non validé cliniquement",
        0.55,
        7.15,
        12.1,
        0.22,
        9,
        "muted",
    )
    return current


def bullets(current, items, y=1.75, size=19, gap=0.78):
    for index, item in enumerate(items):
        top = y + index * gap
        shape(current, 0.9, top + 0.06, 0.12, 0.12, "teal")
        add_text(current, item, 1.18, top, 11.2, gap - 0.08, size)


def card(current, x, y, width, height, heading, body, accent="teal"):
    shape(current, x, y, width, height, "white")
    shape(current, x, y, 0.08, height, accent, rounded=False)
    add_text(current, heading, x + 0.22, y + 0.18, width - 0.42, 0.42, 17, accent, True)
    add_text(current, body, x + 0.22, y + 0.68, width - 0.42, height - 0.82, 15)


PRESENTATION = Presentation()
PRESENTATION.slide_width = Inches(13.333)
PRESENTATION.slide_height = Inches(7.5)
BLANK = PRESENTATION.slide_layouts[6]

# 1 — Cover
current = PRESENTATION.slides.add_slide(BLANK)
current.background.fill.solid()
current.background.fill.fore_color.rgb = color("navy")
shape(current, 0, 0, 13.333, 0.16, "teal", rounded=False)
add_text(current, "MÉMOIRE DE MASTER  •  M1 ILIA  •  SOUTENANCE 15 MIN", 0.85, 0.75, 9, 0.35, 14, "aqua", True)
add_text(
    current,
    "Fusion multimodale pour\nla détection du cancer du sein",
    0.85,
    1.55,
    11.3,
    1.7,
    34,
    "white",
    True,
)
add_text(
    current,
    "Mammographie CMMD · échographie BUSI · variables cliniques observées",
    0.9,
    3.55,
    11,
    0.5,
    20,
    "aqua",
)
add_text(
    current,
    "Fallou Diouck  ·  Habib Thiaw  ·  Abibatou Sylla",
    0.9,
    5.25,
    10,
    0.4,
    16,
    "white",
)
add_text(
    current,
    "VERSION DE TRAVAIL — résultats exploratoires clairement séparés des résultats finaux",
    0.9,
    6.65,
    11,
    0.3,
    12,
    "amber",
    True,
)

# 2 — Question
current = slide("Question de recherche", "01 · Cadrage")
card(current, 0.8, 1.8, 5.7, 2.1, "Question", "Comment comparer des stratégies de fusion à des baselines sans surestimer leur apport ?")
card(current, 6.8, 1.8, 5.7, 2.1, "Précaution fondamentale", "Les jeux publics retenus ne fournissent pas deux modalités appariées chez les mêmes patientes.", "amber")
add_text(current, "La question testable porte sur un protocole expérimental inter-cohortes, pas sur un bénéfice clinique individuel.", 0.95, 4.55, 11.5, 1, 23, "navy", True)

# 3 — Motivation
current = slide("Pourquoi une fusion ?", "02 · Motivation")
bullets(current, [
    "Mammographie, échographie et clinique contiennent des informations de nature différente.",
    "La fusion se compare à des modèles unimodaux et à une référence sans interaction.",
    "Une comparaison n’est interprétable que si cohortes, labels, splits et seuils sont explicités.",
    "Le projet documente les risques de fuite et l’écart entre prototype et validation clinique.",
], size=20)

# 4 — Datasets
current = slide("Jeux de données et variables réellement disponibles", "03 · Données")
card(current, 0.8, 1.75, 5.75, 2, "CMMD · mammographie", "3 744 images\n1 775 identifiants patient\nÂge et type d’anomalie selon disponibilité")
card(current, 6.8, 1.75, 5.75, 2, "BUSI · échographie", "647 images\nMasques disponibles pour localisation\nIdentifiants patient non fiables dans le pipeline retenu", "blue")
add_text(current, "Total documenté : 4 391 images. Prévalences différentes (~70 % CMMD, ~32 % BUSI) : risque de raccourci de provenance.", 0.95, 4.35, 11.6, 1, 18, "ink", True)
add_text(current, "Écho seule (BUSI) et modèles CMMD ne sont pas des comparaisons sur la même cohorte.", 0.95, 5.65, 11.5, 0.7, 16, "red", True)

# 5 — Pairing
current = slide("Limite structurante : paires virtuelles", "04 · Validité")
card(current, 0.85, 1.75, 3.75, 2.35, "Mammographie", "CMMD\nPatiente A\nLabel cible")
card(current, 4.8, 1.75, 3.75, 2.35, "Échographie", "BUSI\nAutre patiente\nPas de lien individuel", "blue")
card(current, 8.75, 1.75, 3.75, 2.35, "Conséquence", "La fusion ne mesure pas une interaction clinique réelle chez la même patiente.", "amber")
add_text(current, "Formulation à retenir : paires virtuelles inter-cohortes. Ne pas conclure à un gain clinique de fusion patient.", 0.95, 5.05, 11.4, 1, 21, "navy", True)

# 6 — Models
current = slide("Modèles comparés", "05 · Méthodes")
card(current, 0.8, 1.75, 3.75, 1.65, "Baselines", "Mammo seule\nÉcho seule\nClinique seule")
card(current, 4.8, 1.75, 3.75, 1.65, "Référence fusion", "Moyenne des logits\nPas d’interaction explicite", "blue")
card(current, 8.8, 1.75, 3.75, 1.65, "Fusions", "Early · Late\nCross-attention\nCross-attention + AMW", "purple")
add_text(current, "Extracteurs du plan : ResNet18, EfficientNet-B7, ReXNet-200, ViT-B/16 et Swin-Tiny. La comparaison principale prévue fixe ResNet18.", 0.95, 4.05, 11.5, 1.15, 17)
add_text(current, "Plan : 28 entraînements + contrôle par permutation à l’inférence = 29 expériences.", 0.95, 5.55, 11.5, 0.6, 17, "navy", True)

# 7 — Evaluation
current = slide("Protocole final prévu", "06 · Évaluation")
bullets(current, [
    "Séparation groupée par patiente si l’identifiant est fiable ; BUSI conserve une limite explicite.",
    "Hold-out (fold −1) réservé ; folds 0–3 entraînement, fold 4 validation selon le protocole versionné.",
    "Appariement indépendant des labels en validation et test ; cinq graines pour la variabilité d’appariement.",
    "Seuil choisi sur validation ; métriques agrégées par patiente ; DeLong aligné et correction de Holm.",
    "Le notebook conforme reste à exécuter sur un runtime PyTorch/timm.",
], size=19)

# 8 — Status
current = slide("Ce qui a été exécuté — et ce qui ne l’a pas été", "07 · Statut des preuves")
card(current, 0.8, 1.75, 5.7, 2.6, "Traces historiques observées", "Sorties Colab T4, entraînements de plusieurs modèles, métriques/figures Drive et checkpoints listés.")
card(current, 6.8, 1.75, 5.7, 2.6, "Non démontré", "29 expériences complètes, protocole corrigé, hold-out final, métriques finales par patiente, chargement local des checkpoints.", "amber")
add_text(current, "Les résultats historiques sont présentés ci-après uniquement à titre exploratoire.", 0.95, 5.1, 11.4, 0.8, 21, "red", True)

# 9 — Exploratory AUC bar chart, directly from exported values
current = slide("AUC-ROC historiques — résultats exploratoires", "08 · Résultats historiques")
values = [
    ("Moyenne logits", "CMMD", 0.9550, "teal"),
    ("Cross-attention + AMW", "CMMD", 0.9354, "purple"),
    ("Écho seule", "BUSI", 0.9339, "blue"),
    ("Contrôle aléatoire", "CMMD", 0.7844, "amber"),
    ("Mammo seule", "CMMD", 0.7528, "green"),
    ("Clinique seule", "CMMD", 0.7462, "red"),
]
for index, (name, cohort, value, accent) in enumerate(values):
    top = 1.72 + index * 0.68
    add_text(current, name, 0.85, top, 2.35, 0.35, 14, "ink", True)
    add_text(current, cohort, 3.18, top, 1, 0.32, 11, "muted")
    shape(current, 4.15, top + 0.02, 6.15, 0.28, "line")
    shape(current, 4.15, top + 0.02, 6.15 * value, 0.28, accent)
    add_text(current, f"{value:.4f}", 10.45, top - 0.02, 1.15, 0.36, 16, accent, True)
add_text(current, "n documenté : BUSI 112 ; CMMD 634. Exports historiques, mesures par image, sans hold-out indépendant validé.", 0.9, 6, 11.6, 0.72, 13, "red", True)
add_text(current, "Cohortes différentes : les barres ne constituent pas une comparaison patient-à-patient.", 0.9, 6.67, 11.6, 0.35, 12, "muted")

# 10 — Caveats
current = slide("Pourquoi les chiffres ne sont pas définitifs", "09 · Audit")
bullets(current, [
    "Paires de validation historiques choisies en utilisant le diagnostic : fuite de cible dans la construction d’entrée.",
    "Le hold-out a été exclu des sorties historiques consultées ; aucune évaluation finale indépendante n’est montrée.",
    "DeLong vérifie seulement la longueur des vecteurs dans le code historique ; certaines AUC exportées sont désalignées.",
    "Les métriques historiques sont par image ; les seuils de Youden sont calculés sur l’échantillon évalué.",
    "AUC clinique exploratoire 0,7462 à investiguer (âge/type d’anomalie).",
], size=17, gap=0.76)

# 11 — Explainability
current = slide("Explicabilité : signal faible, analyse incomplète", "10 · Interprétabilité")
card(current, 0.85, 1.75, 3.7, 2.25, "Grad-CAM · Drive", "IoU moyen : 0,1101\nMédiane : 0\nPointing game : 0,2232\n112 images selon le résumé")
card(current, 4.8, 1.75, 3.7, 2.25, "À réconcilier", "Une sortie notebook rapporte IoU 0,1029 sur 112 images. Version/export à identifier.", "amber")
card(current, 8.75, 1.75, 3.7, 2.25, "SHAP", "Échec rapporté : dimensions de poids incompatibles avec la matrice. Aucune attribution valide.", "red")
add_text(current, "Les exemples Grad-CAM Drive sont hétérogènes ; ne sélectionner pas seulement les cas visuellement favorables.", 0.95, 4.75, 11.4, 1, 19, "navy", True)
add_text(current, "Résultat final : [nouvelle mesure + figure provenant du checkpoint vérifié]", 0.95, 6, 11.4, 0.5, 14, "muted")
add_text(current, "PNG Drive non intégré ici : version de la figure à rapprocher du résumé de localisation.", 0.95, 6.48, 11.4, 0.36, 12, "red", True)

# 12 — Interface
current = slide("Prototype d’interface : statut et garde-fou", "11 · Démonstration")
card(current, 0.85, 1.75, 5.7, 2.2, "Disponible", "Interface minimale : import et aperçu d’image, état du moteur clairement visible.")
card(current, 6.8, 1.75, 5.7, 2.2, "Bloqué volontairement", "Aucun score ni carte d’explication tant qu’un checkpoint n’est pas validé et chargé.", "amber")
add_text(current, "Aucun checkpoint n’est présent localement ; PyTorch et timm sont absents de l’environnement vérifié.", 0.95, 4.55, 11.3, 0.8, 18, "red", True)
add_text(current, "Prototype de recherche — non validé cliniquement — ne pas utiliser pour le diagnostic.", 0.95, 5.8, 11.4, 0.7, 18, "navy", True)

# 13 — Limits
current = slide("Limites à discuter sans détour", "12 · Discussion")
bullets(current, [
    "Aucune paire réelle mammo–écho ; résultats inter-cohortes uniquement.",
    "BUSI ne permet pas une garantie de split patient avec les identifiants du pipeline.",
    "Prévalences et distributions différentes ; possible apprentissage d’indices de provenance.",
    "Mesures historiques invalides pour conclure sur la généralisation ou l’apport clinique.",
    "Pas de validation prospective, de lecture radiologue ou de dispositif médical.",
], size=18, gap=0.78)

# 14 — Next steps
current = slide("Étapes de validation avant version finale", "13 · Suite")
bullets(current, [
    "Inspecter checkpoint, métadonnées, architecture, taille/normalisation d’entrée et empreinte.",
    "Charger strictement le modèle en environnement compatible ; conserver les erreurs et logs.",
    "Exécuter baselines puis fusions prévues ; vérifier reprise, effectifs, labels et patient IDs.",
    "Évaluer une seule fois le hold-out, agrégé par patiente ; produire ROC/PR, IC et DeLong + Holm.",
    "Actualiser tableaux, rapport, présentation et démonstration depuis les mêmes exports vérifiés.",
], size=18, gap=0.78)

# 15 — Closing
current = PRESENTATION.slides.add_slide(BLANK)
current.background.fill.solid()
current.background.fill.fore_color.rgb = color("navy")
shape(current, 0, 0, 13.333, 0.14, "teal", rounded=False)
add_text(current, "CONCLUSION", 0.85, 0.7, 5, 0.3, 12, "aqua", True)
add_text(current, "Un protocole honnête vaut mieux\nqu’une performance non vérifiée.", 0.85, 1.55, 11.5, 1.45, 30, "white", True)
add_text(
    current,
    "Le projet a produit des traces expérimentales et un pipeline ; les chiffres historiques restent exploratoires.\nLa validation finale et le contrôle des checkpoints sont les prochaines preuves nécessaires.",
    0.9,
    3.65,
    11.3,
    1.2,
    19,
    "aqua",
)
add_text(current, "Merci · Questions", 0.9, 5.75, 8, 0.45, 21, "white", True)
add_text(current, "[Résultats finaux à insérer ici uniquement après validation]", 0.9, 6.55, 11, 0.3, 12, "amber", True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
PRESENTATION.save(OUTPUT)
print(f"Created {OUTPUT} ({len(PRESENTATION.slides)} slides)")
