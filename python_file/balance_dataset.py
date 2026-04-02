"""
Rééchantillonnage du dataset par tranche horaire.

Objectif : réduire le déséquilibre tout en conservant la réalité du trafic.
Stratégie : lissage par racine carrée des effectifs réels, puis sous-échantillonnage.

  Période sur-représentée → plafonnée
  Période sous-représentée → conservée intégralement

Les proportions cibles sont proportionnelles à sqrt(effectif_réel), ce qui
preserve l'ordre naturel (Pointe soir > Pointe matin > Creuse …)
mais réduit les écarts extrêmes (ratio max configurable via --ratio_max).

CLI :
  python balance_dataset.py [--csv PATH] [--output PATH]
                            [--ratio_max N] [--col heure_tranche|periode_journee]
                            [--seed N] [--dry-run]
"""

import argparse
import sys
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# Ordre d'affichage des périodes
# ─────────────────────────────────────────────
ORDRE_PERIODES = [
    "Nuit", "Pointe matin", "Creuse matin",
    "Méridienne", "Creuse après-midi", "Pointe soir", "Soirée",
]


def _cibles_lissees(counts: pd.Series, ratio_max: float) -> pd.Series:
    """
    Calcule les effectifs cibles par classe en lissant via sqrt.

    1. Proportions « naturelles » lissées = sqrt(effectif réel)
    2. On normalise pour que la somme reste ≤ sum(counts)
    3. On applique le ratio_max : aucune classe ne dépasse ratio_max × min_cible
    4. Les cibles ne peuvent jamais dépasser l'effectif réel (sous-échantillonnage seul)
    """
    sqrt_counts = np.sqrt(counts)

    # Normalisation pour conserver le total
    facteur = counts.sum() / sqrt_counts.sum()
    cibles = (sqrt_counts * facteur).round().astype(int)

    # Plafond ratio_max
    min_cible = cibles.min()
    plafond = int(min_cible * ratio_max)
    cibles = cibles.clip(upper=plafond)

    # On ne peut pas sur-échantillonner : cible ≤ effectif réel
    cibles = cibles.clip(upper=counts)

    return cibles


def balance(
    df: pd.DataFrame,
    col: str = "heure_tranche",
    ratio_max: float = 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Retourne un DataFrame rééchantillonné selon les effectifs lissés."""
    if col not in df.columns:
        raise ValueError(f"Colonne '{col}' absente du DataFrame.")

    counts = df[col].value_counts().sort_index()
    cibles = _cibles_lissees(counts, ratio_max)

    parts = []
    for classe, n_cible in cibles.items():
        groupe = df[df[col] == classe]
        parts.append(groupe.sample(n=n_cible, random_state=seed))

    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def afficher_rapport(
    counts_avant: pd.Series,
    counts_apres: pd.Series,
    col: str,
) -> None:
    """Affiche un tableau comparatif avant / après."""
    if col == "periode_journee":
        idx = [p for p in ORDRE_PERIODES if p in counts_avant.index]
        # ajoute les valeurs inconnues à la fin
        idx += [p for p in counts_avant.index if p not in idx]
    else:
        idx = sorted(counts_avant.index)

    total_av = counts_avant.sum()
    total_ap = counts_apres.sum()

    print(f"\n{'─'*60}")
    print(f"  Rééchantillonnage — colonne : {col}")
    print(f"{'─'*60}")
    header = f"  {'Classe':<22} {'Avant':>8} {'%':>6}  {'Après':>8} {'%':>6}  {'Δ':>7}"
    print(header)
    print(f"{'─'*60}")

    for classe in idx:
        av = counts_avant.get(classe, 0)
        ap = counts_apres.get(classe, 0)
        pct_av = 100 * av / total_av if total_av else 0
        pct_ap = 100 * ap / total_ap if total_ap else 0
        delta = ap - av
        print(f"  {str(classe):<22} {av:>8,} {pct_av:>5.1f}%  {ap:>8,} {pct_ap:>5.1f}%  {delta:>+7,}")

    print(f"{'─'*60}")
    print(f"  {'TOTAL':<22} {total_av:>8,} {'100%':>6}  {total_ap:>8,} {'100%':>6}  {total_ap - total_av:>+7,}")
    print(f"{'─'*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rééchantillonnage équilibré du dataset par tranche horaire.",
    )
    parser.add_argument(
        "--csv",
        default="dataset_predictions/dataset_ml.csv",
        help="CSV source (défaut : dataset_predictions/dataset_ml.csv)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Chemin de sortie (défaut : <nom_source>_balanced.csv)",
    )
    parser.add_argument(
        "--col",
        default="heure_tranche",
        choices=["heure_tranche", "periode_journee"],
        help="Colonne sur laquelle équilibrer (défaut : heure_tranche)",
    )
    parser.add_argument(
        "--ratio_max",
        type=float,
        default=3.0,
        help="Ratio max entre la classe la plus et la moins représentée (défaut : 3.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire pour la reproductibilité (défaut : 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche le rapport sans écrire le fichier de sortie",
    )
    args = parser.parse_args()

    # ── Lecture ──────────────────────────────────────────────────────────
    print(f"Lecture : {args.csv}")
    try:
        df = pd.read_csv(args.csv, encoding="utf-8-sig", low_memory=False)
    except FileNotFoundError:
        print(f"[ERREUR] Fichier introuvable : {args.csv}", file=sys.stderr)
        sys.exit(1)

    print(f"  → {len(df):,} lignes, {df[args.col].nunique()} valeurs distinctes dans '{args.col}'")

    counts_avant = df[args.col].value_counts().sort_index()

    # ── Rééchantillonnage ─────────────────────────────────────────────────
    df_bal = balance(df, col=args.col, ratio_max=args.ratio_max, seed=args.seed)

    counts_apres = df_bal[args.col].value_counts().sort_index()
    afficher_rapport(counts_avant, counts_apres, args.col)

    if args.dry_run:
        print("[dry-run] Aucun fichier écrit.")
        return

    # ── Écriture ──────────────────────────────────────────────────────────
    if args.output is None:
        base = args.csv.rsplit(".", 1)
        args.output = base[0] + "_balanced." + (base[1] if len(base) > 1 else "csv")

    df_bal.to_csv(args.output, index=False, encoding="utf-8-sig", lineterminator="\n")
    print(f"Fichier sauvegardé : {args.output}")
    print(f"  → {len(df_bal):,} lignes ({100*len(df_bal)/len(df):.1f}% du dataset original)")


if __name__ == "__main__":
    main()
