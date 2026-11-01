// Présentation de soutenance — 16 diapositives, fond blanc, format 15 min.
//   node build_soutenance.js      (les figures sont attendues dans ./fig/)
const pptxgen = require("pptxgenjs");

// ---------------------------------------------------------------- système
const PLUM = "5C2438";   // primaire — dominant
const ROSE = "A8536A";   // secondaire
const INK = "1F1A1C";    // texte
const MUTE = "8A7C82";   // texte secondaire
const HAIR = "E8E2E4";   // filets
const TINT = "FAF6F7";   // fond de carte, à peine teinté
const WASH = "F4ECEF";   // numéros filigranes
const WHITE = "FFFFFF";
const HF = "Cambria", BF = "Calibri";

const W = 13.33, H = 7.5, M = 0.8, CW = W - 2 * M;

const p = new pptxgen();
p.layout = "LAYOUT_WIDE";
p.author = "Fallou DIOUCK, Habib THIAW, Abibatou SYLLA";
p.title = "Fusion multi-modale — Cancer du sein";

// ---------------------------------------------------------------- helpers
let SEC = 0;

function slide(kicker, title, opts = {}) {
  const s = p.addSlide();
  if (opts.num !== false) {
    SEC += 1;
    s.addText(String(SEC).padStart(2, "0"), { x: W - 2.35, y: 0.28, w: 1.6, h: 1.5,
      align: "right", fontFace: HF, fontSize: 76, bold: true, color: WASH, margin: 0 });
  }
  if (kicker) s.addText(kicker.toUpperCase(), { x: M, y: 0.62, w: CW - 2, h: 0.26,
    fontFace: BF, fontSize: 10.5, bold: true, color: ROSE, charSpacing: 2.4, margin: 0 });
  s.addText(title, { x: M, y: 0.9, w: opts.tw || CW - 2.0, h: 0.9, fontFace: HF,
    fontSize: opts.ts || 30, bold: true, color: PLUM, margin: 0 });
  return s;
}

const rule = (s, y, x = M, w = CW) =>
  s.addShape(p.ShapeType.rect, { x, y, w, h: 0.012, fill: { color: HAIR } });

function panel(s, x, y, w, h, tint) {
  s.addShape(p.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.04,
    fill: { color: tint ? TINT : WHITE }, line: { color: HAIR, width: 1 } });
}

function marker(s, x, y, n, c) {
  s.addShape(p.ShapeType.rect, { x, y, w: 0.26, h: 0.26, fill: { color: c || PLUM } });
  s.addText(String(n), { x: x - 0.02, y: y - 0.02, w: 0.3, h: 0.3, align: "center",
    valign: "middle", fontFace: BF, fontSize: 11, bold: true, color: WHITE, margin: 0 });
}

function metric(s, x, y, w, val, lab, c) {
  s.addText(val, { x, y, w, h: 0.8, fontFace: HF, fontSize: 38, bold: true,
    color: c || PLUM, margin: 0 });
  s.addText(lab, { x, y: y + 0.78, w, h: 0.6, fontFace: BF, fontSize: 11,
    color: MUTE, lineSpacing: 15, margin: 0 });
}

const TB = { fontFace: BF, fontSize: 11.5, color: INK, valign: "middle" };
const th = t => ({ text: t, options: { bold: true, color: PLUM, fill: { color: TINT }, fontSize: 11 } });
const foot = (s, t) => s.addText(t, { x: M, y: H - 0.78, w: CW, h: 0.35, fontFace: BF,
  fontSize: 11.5, italic: true, color: ROSE, margin: 0 });

// ============================================================== 01 TITRE
{
  const s = p.addSlide();
  s.addText("ÉCOLE POLYTECHNIQUE DE THIÈS   ·   MASTER 1 ILIA", { x: M, y: 1.15,
    w: CW, h: 0.3, fontFace: BF, fontSize: 10.5, bold: true, color: ROSE,
    charSpacing: 2.4, margin: 0 });
  s.addText("Fusion multi-modale\npar apprentissage automatique", { x: M, y: 1.85,
    w: CW - 1.2, h: 1.9, fontFace: HF, fontSize: 42, bold: true, color: PLUM,
    lineSpacing: 52, margin: 0 });
  s.addText("Détection précoce et interprétation du cancer du sein", { x: M, y: 3.85,
    w: CW - 1.2, h: 0.5, fontFace: HF, fontSize: 19, color: INK, margin: 0 });
  s.addText("Mammographie   ·   Échographie   ·   Données cliniques", { x: M, y: 4.38,
    w: CW, h: 0.35, fontFace: BF, fontSize: 13, italic: true, color: MUTE, margin: 0 });
  rule(s, 5.05, M, 3.2);
  s.addText("Fallou DIOUCK   ·   Habib THIAW   ·   Abibatou SYLLA", { x: M, y: 5.35,
    w: CW, h: 0.32, fontFace: BF, fontSize: 13.5, bold: true, color: INK, margin: 0 });
  s.addText("Encadrants  ·  Pr. GUISSE   Pr. NGOM   Pr. DIOUF   Pr. WADE", { x: M,
    y: 5.75, w: CW, h: 0.32, fontFace: BF, fontSize: 11.5, color: MUTE, margin: 0 });
  s.addNotes("Deux temps : le cadre de recherche, puis notre état d'avancement.");
}

// ============================================================== 02 CONTEXTE
{
  const s = slide("Contexte", "Le dépistage bute sur deux limites");
  metric(s, M, 2.15, 3.3, "> 90 %", "de survie à cinq ans si le cancer\nest détecté à un stade initial");
  metric(s, M + 4.0, 2.15, 3.3, "Seins denses", "la sensibilité de la mammographie\ny diminue nettement", ROSE);
  metric(s, M + 8.0, 2.15, 3.4, "Mono-modal", "la plupart des systèmes exploitent\nune seule source d'information", ROSE);
  rule(s, 3.9);
  const L = [
    ["Mammographie", "Examen de référence : masses non palpables, distorsions, microcalcifications. Sensibilité réduite sur seins denses."],
    ["Échographie", "Complément utile sur seins denses, mais faible contraste et forte variabilité inter-opérateur."],
    ["Données cliniques", "Informatives et peu coûteuses — pourtant quasiment jamais jointes à l'imagerie."],
  ];
  L.forEach(([t, d], i) => {
    const y = 4.25 + i * 0.82;
    marker(s, M, y + 0.09, i + 1);
    s.addText(t, { x: M + 0.5, y, w: 2.6, h: 0.4, fontFace: HF, fontSize: 15,
      bold: true, color: PLUM, valign: "middle", margin: 0 });
    s.addText(d, { x: M + 3.2, y, w: CW - 3.2, h: 0.45, fontFace: BF, fontSize: 12.5,
      color: INK, valign: "middle", margin: 0 });
  });
}

// ============================================================== 03 VERROU
{
  const s = slide("Le verrou central", "Aucun jeu public n'apparie les deux modalités");
  panel(s, M, 2.1, CW, 1.3, true);
  s.addText("« Bien que la combinaison de la mammographie et de l'échographie soit une pratique courante en dépistage, il n'existe pas de jeu de données public contenant les lésions correspondantes dans les deux modalités. »",
    { x: M + 0.4, y: 2.3, w: CW - 0.8, h: 0.95, fontFace: HF, fontSize: 13.5,
      italic: true, color: PLUM, lineSpacing: 20, margin: 0 });

  const C = [
    ["La cross-attention suppose une correspondance", "Elle met à jour chaque modalité selon les autres. Encore faut-il que les deux images montrent la même lésion chez la même femme."],
    ["Nos paires sont virtuelles", "Nous associons deux patientes différentes partageant le même diagnostic. Aucune interaction réelle n'existe dans les données."],
    ["D'où un modèle de référence", "Chaque stratégie est comparée à une moyenne pondérée des prédictions, qui ne modélise aucune interaction."],
  ];
  C.forEach(([t, d], i) => {
    const x = M + i * (CW / 3 + 0.05), w = CW / 3 - 0.28;
    marker(s, x, 3.85, i + 1, i === 2 ? ROSE : PLUM);
    s.addText(t, { x, y: 4.28, w, h: 0.75, fontFace: HF, fontSize: 14.5, bold: true,
      color: PLUM, lineSpacing: 19, margin: 0 });
    s.addText(d, { x, y: 5.1, w, h: 1.3, fontFace: BF, fontSize: 12, color: MUTE,
      lineSpacing: 17, valign: "top", margin: 0 });
  });
  foot(s, "Les deux seules études de fusion mammographie + échographie reposent sur des données hospitalières privées — 92 et 790 patientes.");
}

// ============================================================== 04 QUESTION
{
  const s = slide("Problématique", "Question de recherche", { ts: 26 });
  s.addText("Comment combiner les informations issues de la mammographie, de l'échographie et des données cliniques dans un modèle profond pour obtenir une prédiction plus fiable et interprétable ?",
    { x: M, y: 2.3, w: CW - 1.6, h: 2.3, fontFace: HF, fontSize: 27, bold: true,
      color: INK, lineSpacing: 40, margin: 0 });
  rule(s, 5.0, M, 3.2);
  const K = [["Fiable", "meilleure sensibilité et spécificité"],
             ["Interprétable", "compréhensible par un radiologue"]];
  K.forEach(([t, d], i) => {
    const x = M + i * 5.4;
    s.addText(t, { x, y: 5.35, w: 5.0, h: 0.4, fontFace: HF, fontSize: 19, bold: true,
      color: PLUM, margin: 0 });
    s.addText(d, { x, y: 5.78, w: 5.0, h: 0.4, fontFace: BF, fontSize: 12.5, color: MUTE, margin: 0 });
  });
}

// ============================================================== 05 ETAT DE L'ART
{
  const s = slide("État de l'art", "Notre positionnement");
  s.addTable([
    [th("Étude"), th("Modalités"), th("Effectif"), th("Explicabilité"), th("Données")],
    ["Alswilem & Pacal (2025)", "1 — mammographie", "3 383 images", "Non", "Publiques"],
    ["Latha et al. (2024)", "1 — échographie", "780 images", "Grad-CAM qualitatif", "Publiques"],
    ["Chen et al. (2025)", "2 — mammo + écho", "790 patientes", "Grad-CAM qualitatif", "Privées"],
    ["Rashedi et al. (2026)", "3 — avec âge", "92 patientes", "Non", "Privées"],
    [{ text: "Notre projet", options: { bold: true, color: PLUM, fill: { color: TINT } } },
     { text: "3 — avec clinique", options: { bold: true, color: PLUM, fill: { color: TINT } } },
     { text: "4 278 patientes", options: { bold: true, color: PLUM, fill: { color: TINT } } },
     { text: "Grad-CAM mesuré + SHAP", options: { bold: true, color: PLUM, fill: { color: TINT } } },
     { text: "Publiques", options: { bold: true, color: PLUM, fill: { color: TINT } } }],
  ], { x: M, y: 2.2, w: CW, colW: [2.9, 2.5, 2.1, 2.63, 1.5], rowH: 0.5,
       border: { type: "solid", color: HAIR, pt: 1 }, ...TB });

  const N = [["Effectif", "Un ordre de grandeur au-dessus des études de fusion existantes"],
             ["Explicabilité", "Quantifiée par mesure d'IoU, non plus illustrée"],
             ["Reproductibilité", "Données publiques, code ouvert, garde-fous testés"]];
  N.forEach(([t, d], i) => {
    const x = M + i * (CW / 3 + 0.05), w = CW / 3 - 0.28;
    s.addText(t, { x, y: 5.6, w, h: 0.35, fontFace: HF, fontSize: 14, bold: true, color: PLUM, margin: 0 });
    s.addText(d, { x, y: 5.98, w, h: 0.6, fontFace: BF, fontSize: 11.5, color: MUTE,
      lineSpacing: 15, margin: 0 });
  });
}

// ============================================================== 06 DONNEES
{
  const s = slide("Sources de données", "Six jeux publics, tous librement accessibles");
  s.addTable([
    [th("Jeu"), th("Modalité"), th("Images"), th("Patientes"), th("Malignes"), th("Clinique réelle")],
    ["CBIS-DDSM", "Mammographie", "2 857", "1 460", "44,7 %", "Densité BI-RADS"],
    ["CMMD", "Mammographie", "3 744", "1 775", "70,3 %", "Âge, type de lésion"],
    ["MIAS", "Mammographie", "115", "108", "45,2 %", "—"],
    ["BUSI", "Échographie", "647", "647", "32,5 %", "—"],
    ["BrEaST", "Échographie", "252", "252", "38,9 %", "Âge, BI-RADS"],
    ["BUS-UCLM", "Échographie", "264", "36", "34,1 %", "—"],
    [{ text: "Total", options: { bold: true, color: PLUM, fill: { color: TINT } } },
     { text: "", options: { fill: { color: TINT } } },
     { text: "7 879", options: { bold: true, color: PLUM, fill: { color: TINT } } },
     { text: "4 278", options: { bold: true, color: PLUM, fill: { color: TINT } } },
     { text: "55,3 %", options: { bold: true, color: PLUM, fill: { color: TINT } } },
     { text: "3 955 lignes", options: { bold: true, color: PLUM, fill: { color: TINT } } }],
  ], { x: M, y: 2.15, w: CW, colW: [2.0, 2.2, 1.6, 1.8, 1.8, 2.23], rowH: 0.44,
       border: { type: "solid", color: HAIR, pt: 1 }, ...TB });
  foot(s, "INbreast, UDIAT et BUI — accès sur demande — ont été remplacés par CMMD, BrEaST et BUS-UCLM : effectif multiplié par quinze en test externe, et l'âge réel de 1 775 patientes en plus.");
}

// ============================================================== 07 METHODO
{
  const s = slide("Méthodologie", "Deux règles qui conditionnent la validité");
  const G = [
    ["Découpage au niveau patiente", "Plusieurs clichés par patiente dans CBIS-DDSM, CMMD et BUS-UCLM.",
     "Un découpage par image placerait deux vues d'une même femme de part et d'autre du test : le modèle mémoriserait la patiente au lieu d'apprendre la lésion.",
     "StratifiedGroupKFold sur l'identifiant patiente"],
    ["Variables cliniques réelles seulement", "Âge, densité BI-RADS et type de lésion — mesurés sur la patiente.",
     "Générer des antécédents ou un statut BRCA conditionnellement au diagnostic reviendrait à apprendre le générateur plutôt que la pathologie.",
     "Un modèle clinique seul sert d'instrument de contrôle"],
  ];
  G.forEach(([t, ctx, why, how], i) => {
    const x = M + i * (CW / 2 + 0.14), w = CW / 2 - 0.14;
    panel(s, x, 2.15, w, 3.9);
    marker(s, x + 0.35, 2.5, i + 1);
    s.addText(t, { x: x + 0.35, y: 2.92, w: w - 0.7, h: 0.45, fontFace: HF,
      fontSize: 16.5, bold: true, color: PLUM, margin: 0 });
    s.addText(ctx, { x: x + 0.35, y: 3.4, w: w - 0.7, h: 0.5, fontFace: BF,
      fontSize: 12.5, color: INK, lineSpacing: 17, margin: 0 });
    s.addText(why, { x: x + 0.35, y: 3.98, w: w - 0.7, h: 1.1, fontFace: BF,
      fontSize: 12, color: MUTE, lineSpacing: 17, valign: "top", margin: 0 });
    s.addShape(p.ShapeType.roundRect, { x: x + 0.35, y: 5.2, w: w - 0.7, h: 0.55,
      rectRadius: 0.04, fill: { color: TINT } });
    s.addText(how, { x: x + 0.5, y: 5.2, w: w - 1.0, h: 0.55, fontFace: BF,
      fontSize: 11.5, bold: true, color: PLUM, valign: "middle", margin: 0 });
  });
  foot(s, "Chacune est protégée par un test automatique exécuté avant tout entraînement — 61 tests au total.");
}

// ============================================================== 08 FUSION
{
  const s = slide("Stratégies comparées", "Quatre stratégies de fusion, une référence");
  const F = [
    ["Réf.", "Moyenne des logits", "Prédictions indépendantes combinées par une somme pondérée apprise. Aucune interaction modélisée.", true],
    ["a", "Concaténation précoce", "Fusion au niveau pixel, extracteur unique. Borne basse attendue.", false],
    ["b", "Concaténation tardive", "Embeddings concaténés avant le classifieur. Approche de Chen et al.", false],
    ["c", "Cross-attention", "Attention multi-têtes entre modalités. Approche de Rashedi et al.", false],
    ["d", "Cross-attention + AMW", "Pondération adaptative par cas selon la qualité de chaque modalité.", false],
  ];
  F.forEach(([n, t, d, ref], i) => {
    const y = 2.2 + i * 0.82;
    if (ref) s.addShape(p.ShapeType.roundRect, { x: M, y: y - 0.08, w: CW, h: 0.74,
      rectRadius: 0.04, fill: { color: TINT } });
    s.addText(n, { x: M + 0.25, y, w: 0.8, h: 0.58, fontFace: BF, fontSize: 12,
      bold: true, color: ref ? PLUM : ROSE, valign: "middle", margin: 0 });
    s.addText(t, { x: M + 1.15, y, w: 3.5, h: 0.58, fontFace: HF, fontSize: 15,
      bold: true, color: PLUM, valign: "middle", margin: 0 });
    s.addText(d, { x: M + 4.8, y, w: CW - 5.1, h: 0.58, fontFace: BF, fontSize: 12,
      color: MUTE, valign: "middle", margin: 0 });
    if (i < F.length - 1) rule(s, y + 0.68);
  });
  foot(s, "Mêmes extracteurs pour toutes les stratégies : seule la combinaison varie. Tout écart d'AUC passe par le test de DeLong.");
}

// ============================================================== 09 PROTOCOLE
{
  const s = slide("Protocole d'évaluation", "Quatre comparaisons, et une question clinique");
  const C = [
    ["Fusion contre meilleure modalité seule", "Sans ce gain, la fusion n'a pas d'intérêt."],
    ["Attention contre moyenne des logits", "La comparaison centrale du projet."],
    ["AMW contre cross-attention simple", "Les poids par cas révèlent la modalité dominante."],
    ["Appariement aléatoire", "Test d'intégrité : le modèle utilise-t-il vraiment les deux modalités ?"],
  ];
  C.forEach(([t, d], i) => {
    const y = 2.2 + i * 0.78;
    marker(s, M, y + 0.1, i + 1, i === 1 ? PLUM : ROSE);
    s.addText(t, { x: M + 0.5, y, w: 5.3, h: 0.4, fontFace: HF, fontSize: 14.5,
      bold: true, color: PLUM, valign: "middle", margin: 0 });
    s.addText(d, { x: M + 5.9, y, w: CW - 5.9, h: 0.4, fontFace: BF, fontSize: 12,
      color: MUTE, valign: "middle", margin: 0 });
  });
  panel(s, M, 5.45, CW, 1.2, true);
  s.addText("Au-delà de l'AUC : « à sensibilité garantie, quelle proportion de biopsies inutiles évite-t-on ? »",
    { x: M + 0.4, y: 5.62, w: CW - 0.8, h: 0.4, fontFace: HF, fontSize: 15.5,
      bold: true, color: PLUM, margin: 0 });
  s.addText("Le seuil 0,5, hérité de la fonction de décision, n'a pas de sens clinique. En dépistage on fixe d'abord la sensibilité, puis on lit la spécificité obtenue.",
    { x: M + 0.4, y: 6.03, w: CW - 0.8, h: 0.45, fontFace: BF, fontSize: 12, color: MUTE, margin: 0 });
}

// ============================================================== 10 ARCHITECTURE
{
  const s = slide("Conception", "Architecture du système");
  const top = [["Mammographie", "CC / MLO"], ["Échographie", "B-mode"], ["Clinique C0", "âge, densité, lésion"]];
  top.forEach(([t, d], i) => {
    const x = M + i * (CW / 3 + 0.05), w = CW / 3 - 0.28;
    panel(s, x, 2.15, w, 0.82, true);
    s.addText(t, { x: x + 0.22, y: 2.26, w: w - 0.44, h: 0.32, fontFace: HF,
      fontSize: 14, bold: true, color: PLUM, margin: 0 });
    s.addText(d, { x: x + 0.22, y: 2.58, w: w - 0.44, h: 0.3, fontFace: BF,
      fontSize: 11, color: MUTE, margin: 0 });
  });
  ["Encodeur image", "Encodeur image", "Encodeur MLP"].forEach((t, i) => {
    const x = M + i * (CW / 3 + 0.05), w = CW / 3 - 0.28;
    panel(s, x, 3.15, w, 0.58);
    s.addText(t, { x: x + 0.22, y: 3.15, w: w - 0.44, h: 0.58, fontFace: BF,
      fontSize: 12, color: INK, valign: "middle", margin: 0 });
  });
  s.addShape(p.ShapeType.roundRect, { x: M, y: 3.95, w: CW, h: 0.68, rectRadius: 0.04,
    fill: { color: PLUM } });
  s.addText("Fusion multi-modale  —  cross-attention multi-têtes + pondération adaptative (AMW)",
    { x: M, y: 3.95, w: CW, h: 0.68, align: "center", valign: "middle", fontFace: BF,
      fontSize: 13, bold: true, color: WHITE, margin: 0 });
  s.addShape(p.ShapeType.roundRect, { x: M + 2.6, y: 4.82, w: CW - 5.2, h: 0.58,
    rectRadius: 0.04, fill: { color: ROSE } });
  s.addText("Classifieur  —  score de malignité [0–1] + confiance", { x: M + 2.6,
    y: 4.82, w: CW - 5.2, h: 0.58, align: "center", valign: "middle", fontFace: BF,
    fontSize: 12, bold: true, color: WHITE, margin: 0 });
  ["Grad-CAM validé par IoU", "Prédiction", "SHAP — variables cliniques"].forEach((t, i) => {
    const x = M + i * (CW / 3 + 0.05), w = CW / 3 - 0.28;
    panel(s, x, 5.6, w, 0.58, true);
    s.addText(t, { x: x + 0.22, y: 5.6, w: w - 0.44, h: 0.58, fontFace: BF,
      fontSize: 11.5, color: PLUM, valign: "middle", margin: 0 });
  });
  s.addText("Interface radiologue (React.js)   ·   API FastAPI — /predict et /explain",
    { x: M, y: 6.4, w: CW, h: 0.4, align: "center", fontFace: BF, fontSize: 11.5,
      color: MUTE, margin: 0 });
}

// ============================================================== 11 AVANCEMENT
{
  const s = slide("Avancement", "Où nous en sommes");
  const PH = [
    ["Phase 0", "Cadrage et socle technique", "Terminée", 100],
    ["Phase 1", "Pipeline de données et manifeste unifié", "Terminée", 100],
    ["Phase 2", "Baselines mono-modaux", "Terminée", 100],
    ["Phase 3", "Fusion multi-modale", "En cours", 60],
    ["Phase 4", "Explicabilité et généralisation", "En cours", 45],
    ["Phase 5", "API et interface radiologue", "À venir", 0],
    ["Phase 6", "Validation clinique et rapport", "À venir", 0],
  ];
  PH.forEach(([ph, t, st, pct], i) => {
    const y = 2.2 + i * 0.62, done = pct === 100, prog = pct > 0 && pct < 100;
    const c = done ? PLUM : prog ? ROSE : HAIR;
    s.addShape(p.ShapeType.rect, { x: M, y: y + 0.12, w: 0.22, h: 0.22, fill: { color: c } });
    s.addText(ph, { x: M + 0.45, y, w: 1.15, h: 0.45, fontFace: BF, fontSize: 11.5,
      bold: true, color: MUTE, valign: "middle", margin: 0 });
    s.addText(t, { x: M + 1.65, y, w: 6.0, h: 0.45, fontFace: BF, fontSize: 13.5,
      color: INK, valign: "middle", margin: 0 });
    s.addText(st, { x: M + 7.8, y, w: 1.5, h: 0.45, fontFace: BF, fontSize: 11,
      bold: true, color: done ? PLUM : prog ? ROSE : MUTE, valign: "middle", margin: 0 });
    s.addShape(p.ShapeType.rect, { x: M + 9.5, y: y + 0.19, w: 2.13, h: 0.08, fill: { color: HAIR } });
    if (pct > 0) s.addShape(p.ShapeType.rect, { x: M + 9.5, y: y + 0.19,
      w: 2.13 * pct / 100, h: 0.08, fill: { color: done ? PLUM : ROSE } });
  });
  foot(s, "7 879 images · 4 278 patientes · 61 tests automatiques · aucune fuite patiente détectée.");
}

// ============================================================== 12 RESULTATS
{
  const s = slide("Résultats préliminaires", "BUSI, CMMD et âge réel");
  s.addTable([
    [th("Modèle"), th("AUC"), th("Sens."), th("Spéc."), th("MCC")],
    [{ text: "Fusion — moyenne des logits", options: { bold: true, color: PLUM, fill: { color: TINT } } },
     { text: "0,972", options: { bold: true, color: PLUM, fill: { color: TINT } } },
     { text: "0,925", options: { fill: { color: TINT } } },
     { text: "0,902", options: { fill: { color: TINT } } },
     { text: "0,812", options: { fill: { color: TINT } } }],
    [{ text: "Fusion — cross-attention + AMW", options: { bold: true, color: PLUM, fill: { color: TINT } } },
     { text: "0,960", options: { bold: true, color: PLUM, fill: { color: TINT } } },
     { text: "0,880", options: { fill: { color: TINT } } },
     { text: "0,943", options: { fill: { color: TINT } } },
     { text: "0,784", options: { fill: { color: TINT } } }],
    ["Échographie seule", "0,914", "0,946", "0,773", "0,678"],
    ["Appariement aléatoire", "0,789", "0,798", "0,696", "0,476"],
    ["Mammographie seule", "0,763", "0,659", "0,742", "0,371"],
    ["Clinique seule", "0,749", "0,586", "0,804", "0,361"],
  ], { x: M, y: 2.2, w: 7.0, colW: [3.0, 1.05, 1.0, 1.0, 0.95], rowH: 0.46,
       border: { type: "solid", color: HAIR, pt: 1 }, ...TB, fontSize: 11 });

  s.addImage({ path: "fig/roc_hd.png", x: M + 7.35, y: 2.05, w: 4.28, h: 3.6 });

  s.addText("La fusion apporte : 0,972 contre 0,763 pour la mammographie seule. Le contrôle par appariement aléatoire chute à 0,789 — le modèle utilise bien les deux modalités.",
    { x: M, y: 5.85, w: 7.0, h: 0.85, fontFace: BF, fontSize: 12, color: INK,
      lineSpacing: 17, margin: 0 });
  s.addText("Poids appris : échographie 0,36 · clinique 0,32 · mammographie 0,32. Nous reproduisons la dominance de l'échographie rapportée par Rashedi et al.",
    { x: M + 7.35, y: 5.85, w: 4.28, h: 0.85, fontFace: BF, fontSize: 11, color: MUTE,
      lineSpacing: 15, margin: 0 });
}

// ============================================================== 13 RESULTAT PRINCIPAL
{
  const s = slide("Résultat principal", "La cross-attention ne bat pas une moyenne pondérée", { ts: 28 });
  s.addTable([
    [th(""), th("Moyenne des logits"), th("Cross-attention AMW"), th("Écart"), th("DeLong")],
    [{ text: "Exécution 1", options: { bold: true, color: PLUM } }, "0,9526", "0,9591",
     { text: "+0,0065", options: { bold: true } }, "p = 0,34"],
    [{ text: "Exécution 2", options: { bold: true, color: PLUM } }, "0,9721", "0,9597",
     { text: "−0,0125", options: { bold: true } }, "p = 0,02"],
  ], { x: M, y: 2.25, w: CW, colW: [2.3, 2.9, 2.9, 1.9, 1.63], rowH: 0.52,
       border: { type: "solid", color: HAIR, pt: 1 }, ...TB, fontSize: 12.5 });

  s.addText("Le même code, les mêmes données : d'une exécution à l'autre, le signe de l'écart s'inverse.",
    { x: M, y: 4.15, w: CW - 1.0, h: 0.42, fontFace: HF, fontSize: 17, bold: true,
      color: PLUM, margin: 0 });
  s.addText("La variabilité entre exécutions atteint 0,02 d'AUC — soit davantage que l'écart entre les deux stratégies. Un mécanisme d'attention à plusieurs millions de paramètres n'apporte donc rien de mesurable face à une somme pondérée de trois scalaires.",
    { x: M, y: 4.72, w: CW - 1.0, h: 1.0, fontFace: BF, fontSize: 14, color: INK,
      lineSpacing: 21, margin: 0 });
  rule(s, 5.95, M, 3.2);
  s.addText("C'est une vérification que la littérature sur ces architectures ne fait pratiquement jamais.",
    { x: M, y: 6.15, w: CW - 1.0, h: 0.45, fontFace: HF, fontSize: 15.5, italic: true,
      bold: true, color: ROSE, margin: 0 });
  s.addNotes("Insister : ce n'est pas un échec. Sur des paires virtuelles il n'existe aucune interaction inter-modale à apprendre — nous le démontrons au lieu de le supposer.");
}

// ============================================================== 14 UTILITE CLINIQUE
{
  const s = slide("Lecture clinique", "L'AUC ne dit pas l'utilité clinique");
  s.addImage({ path: "fig/clinique_hd.png", x: M, y: 2.15, w: 6.6, h: 3.97 });

  s.addText("À 95 % de sensibilité", { x: M + 7.1, y: 2.3, w: CW - 7.1, h: 0.4,
    fontFace: HF, fontSize: 19, bold: true, color: PLUM, margin: 0 });
  s.addText("La fusion conserve 78 % de spécificité, quand la mammographie seule tombe à 22 %.",
    { x: M + 7.1, y: 2.78, w: CW - 7.1, h: 0.75, fontFace: BF, fontSize: 13.5,
      color: INK, lineSpacing: 19, margin: 0 });
  rule(s, 3.75, M + 7.1, CW - 7.1);
  s.addText("Ce que cela signifie", { x: M + 7.1, y: 3.95, w: CW - 7.1, h: 0.35,
    fontFace: HF, fontSize: 15, bold: true, color: PLUM, margin: 0 });
  s.addText([
    { text: "À sensibilité fixée — ne pas manquer de cancer — la fusion divise par plus de trois le taux de biopsies inutiles.", options: { bullet: true, breakLine: true } },
    { text: "C'est la métrique qui parle aux radiologues, et celle que la littérature rapporte rarement.", options: { bullet: true } },
  ], { x: M + 7.1, y: 4.35, w: CW - 7.1, h: 1.7, fontFace: BF, fontSize: 12.5,
       color: MUTE, lineSpacing: 18, paraSpaceAfter: 8, valign: "top", margin: 0 });
}

// ============================================================== 15 EXPLICABILITE
{
  const s = slide("Explicabilité mesurée", "Le modèle regarde-t-il la lésion ?");
  s.addImage({ path: "fig/gradcam_hd.png", x: M, y: 2.05, w: 8.1, h: 4.33 });

  metric(s, M + 8.55, 2.1, 3.1, "54 %", "de visée de la lésion\nsur les cas malins");
  metric(s, M + 8.55, 3.62, 3.1, "0,103", "IoU moyen entre carte\net masque annoté", ROSE);
  rule(s, 5.18, M + 8.55, 3.1);
  s.addText("Lecture honnête", { x: M + 8.55, y: 5.36, w: 3.1, h: 0.32, fontFace: HF,
    fontSize: 14, bold: true, color: PLUM, margin: 0 });
  s.addText("Sur les cas bénins la visée tombe à 4 % — et c'est attendu : la carte est calculée sur le logit « malin », or aucune région ne soutient la malignité dans un cas bénin. Limite de la méthode, pas du modèle.",
    { x: M + 8.55, y: 5.74, w: 3.1, h: 1.3, fontFace: BF, fontSize: 10.5, color: MUTE,
      lineSpacing: 14, valign: "top", margin: 0 });
}

// ============================================================== 16 CONCLUSION
{
  const s = slide("En résumé", "Ce que nous établissons, ce qu'il reste");
  const R = [
    ["Un jeu de données constitué", "7 879 images, 4 278 patientes, six sources publiques, aucune fuite au niveau patiente."],
    ["Une contrainte assumée", "Aucun jeu public n'apparie les deux modalités : nos paires sont virtuelles, et nous le mesurons."],
    ["Un résultat méthodologique", "La cross-attention n'apporte rien de mesurable face à une moyenne pondérée — démontré, non supposé."],
    ["Une explicabilité chiffrée", "54 % de visée de la lésion sur les cas malins, au lieu d'une planche d'images."],
  ];
  R.forEach(([t, d], i) => {
    const y = 2.15 + i * 0.9;
    marker(s, M, y + 0.08, i + 1);
    s.addText(t, { x: M + 0.5, y, w: 4.2, h: 0.4, fontFace: HF, fontSize: 15,
      bold: true, color: PLUM, valign: "middle", margin: 0 });
    s.addText(d, { x: M + 4.8, y, w: CW - 4.8, h: 0.45, fontFace: BF, fontSize: 12.5,
      color: INK, valign: "middle", margin: 0 });
  });
  rule(s, 5.95);
  s.addText("Prochaines étapes", { x: M, y: 6.15, w: 3.0, h: 0.35, fontFace: HF,
    fontSize: 14, bold: true, color: PLUM, margin: 0 });
  s.addText("Criblage des cinq architectures  ·  validation croisée à cinq plis  ·  généralisation inter-jeux  ·  API et interface radiologue",
    { x: M + 3.0, y: 6.15, w: CW - 3.0, h: 0.4, fontFace: BF, fontSize: 12,
      color: MUTE, margin: 0 });
}

p.writeFile({ fileName: "Soutenance_Fusion_MultiModale.pptx" })
 .then(f => console.log("écrit :", f));
