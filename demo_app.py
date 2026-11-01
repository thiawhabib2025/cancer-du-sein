"""Streamlit research demo: inspect images without producing predictions."""

from pathlib import Path

import streamlit as st

from src.demo.readiness import decode_image_preview, inspect_runtime


PROJECT_ROOT = Path(__file__).resolve().parent
UPLOADERS = {
    "upload_mammography": "Mammographie",
    "upload_ultrasound": "Échographie",
}
EMPTY_IMAGE_MESSAGES = {
    "Mammographie": "Aucune image de mammographie chargée.",
    "Échographie": "Aucune image d’échographie chargée.",
}

st.set_page_config(
    page_title="Breast Imaging Research Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    .block-container { max-width: 1440px; padding-top: 1.6rem; }
    .research-warning {
        position: sticky;
        top: 0;
        z-index: 999;
        padding: .8rem 1rem;
        margin: 0 0 1.2rem;
        border: 1px solid #c65b61;
        border-left: 6px solid #a63d40;
        border-radius: .55rem;
        background: #fff1f0;
        color: #702b2e;
        font-weight: 750;
        box-shadow: 0 3px 12px rgba(48, 28, 28, .08);
    }
    .hero {
        padding: 1.35rem 1.5rem;
        margin: 0 0 1rem;
        border: 1px solid rgba(100, 116, 139, .2);
        border-radius: .9rem;
        background: linear-gradient(115deg, #f0f7ff 0%, #f7f5ff 58%, #fbfcff 100%);
    }
    .hero-kicker {
        color: #405b83;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .hero h1 { margin: .35rem 0 .45rem; color: #172b4d; }
    .hero p { margin: 0; color: #46566f; font-size: 1.04rem; }
    .small-note { color: #53647b; font-size: .9rem; }
    </style>
    <div class="research-warning" role="alert">
      Prototype de recherche non validé cliniquement — ne pas utiliser pour le diagnostic.
    </div>
    <section class="hero">
      <div class="hero-kicker">EPT · M1 ILIA · Démonstrateur de recherche</div>
      <h1>Imagerie mammaire multimodale</h1>
      <p>Parcours de démonstration local : import, contrôle technique et aperçu des images.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

status = inspect_runtime(PROJECT_ROOT)
with st.sidebar:
    st.header("État du moteur")
    st.error("Inférence indisponible — aucun checkpoint n’a été validé par chargement strict.")
    st.caption(
        "Les fichiers présents ne sont pas assimilés à des modèles validés. "
        "Aucun score ni Grad-CAM n’est produit."
    )
    st.write(f"PyTorch disponible : {'oui' if status['torch_available'] else 'non'}")
    st.write(f"timm disponible : {'oui' if status['timm_available'] else 'non'}")
    st.write(f"Checkpoints locaux : {len(status['checkpoints'])}")
    st.write(
        "Manifeste modèle : "
        + ("présent, non validé" if status["manifest_available"] else "absent")
    )
    if status["checkpoints"]:
        with st.expander("Fichiers détectés (non validés)", expanded=False):
            for name in status["checkpoints"]:
                st.code(name, language=None)
    st.info("L’inventaire et l’inférence réelle restent en attente d’un environnement compatible.")

st.warning(
    "N’importez pas d’images médicales identifiantes ni de données personnelles. "
    "Les images servent uniquement à l’aperçu de cette session et ne sont pas "
    "transmises à un modèle."
)

if "upload_revision" not in st.session_state:
    st.session_state["upload_revision"] = 0

if st.button("Effacer les images", type="secondary"):
    st.session_state["upload_revision"] += 1
    st.rerun()

demo_tab, method_tab, status_tab = st.tabs(
    ["Démonstration visuelle", "Méthodologie & limites", "État technique"]
)

with demo_tab:
    st.subheader("Importer et prévisualiser")
    st.caption(
        "Les deux espaces d’import sont indépendants. Une image peut être chargée "
        "sans déclencher d’analyse."
    )
    mammography_column, ultrasound_column = st.columns(2, gap="large")
    for column, (key, modality) in zip(
        (mammography_column, ultrasound_column), UPLOADERS.items()
    ):
        with column:
            st.markdown(f"#### {modality}")
            uploaded_image = st.file_uploader(
                f"Choisir une image ({modality.lower()})",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                key=f"{key}_{st.session_state['upload_revision']}",
                help="Formats acceptés : PNG, JPEG ou TIFF ; taille maximale : 20 Mio.",
            )
            if uploaded_image is None:
                st.info(EMPTY_IMAGE_MESSAGES[modality])
                continue
            try:
                preview, metadata = decode_image_preview(uploaded_image.getvalue())
            except ValueError as exc:
                st.error(f"Image refusée : {exc}")
                continue

            st.image(preview, caption=f"Aperçu — {modality}", width=600)
            st.caption(
                f"{metadata['format']} · {metadata['width']} × {metadata['height']} px · "
                f"{metadata['mode']} · {len(uploaded_image.getvalue()) / (1024 * 1024):.2f} Mio"
            )
            if metadata["frames"] > 1:
                st.caption("Image multipage : seule la première image est prévisualisée.")

    st.divider()
    st.info(
        "Aucune inférence n’est exécutée. Le bouton d’analyse est volontairement "
        "absent : aucun score, diagnostic ou carte d’explication simulé n’est affiché."
    )

with method_tab:
    st.subheader("Ce que la démonstration permet — et ne permet pas")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Parcours vérifiable")
        st.markdown(
            """
            1. Charger une image anonymisée au format PNG, JPEG ou TIFF.
            2. Contrôler que le fichier est décodable et que ses dimensions sont raisonnables.
            3. Afficher l’aperçu et les propriétés techniques.
            4. Effacer les fichiers pour revenir à l’écran initial.
            """
        )
    with right:
        st.markdown("#### Limites scientifiques")
        st.markdown(
            """
            - Les données CMMD et BUSI ne sont pas des examens appariés des mêmes patientes.
            - Les paires de fusion prévues sont virtuelles et ne prouvent pas un bénéfice clinique individuel.
            - Les identifiants BUSI ne garantissent pas une séparation fiable par patiente.
            - Les résultats historiques restent exploratoires ; aucun résultat final n’est calculé ici.
            """
        )
    st.warning(
        "Cette interface est un prototype logiciel de recherche. Elle ne constitue "
        "ni un dispositif médical ni un outil d’aide au diagnostic."
    )

with status_tab:
    st.subheader("Disponibilité locale")
    st.write(
        "Les poids ci-dessous ont été récupérés localement et leur intégrité "
        "d’archive a été contrôlée lors d’une étape précédente. Leur chargement "
        "strict et une inférence contrôlée ne sont pas validés."
    )
    if status["checkpoints"]:
        st.dataframe(
            [{"Fichier": name, "Statut": "Présent — non validé"} for name in status["checkpoints"]],
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("Aucun fichier de checkpoint local détecté.")

    st.markdown("#### Contrôles encore requis")
    st.markdown(
        """
        - Runtime compatible PyTorch/timm et accès aux fichiers.
        - Architecture et prétraitement associés à chaque checkpoint.
        - Chargement sûr avec correspondance stricte des poids.
        - Inférence contrôlée sur une entrée connue et consignation des erreurs.
        - Évaluation indépendante avant toute présentation de performance.
        """
    )
    st.caption(
        "Le dossier Drive n’est pas requis pour cette démonstration d’aperçu. "
        "La procédure d’inventaire des checkpoints reste en attente."
    )

st.divider()
st.markdown(
    '<p class="small-note">Prototype académique · aucune décision clinique · '
    "ne pas téléverser de données identifiantes.</p>",
    unsafe_allow_html=True,
)
