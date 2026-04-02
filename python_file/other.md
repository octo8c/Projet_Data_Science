
```
def _train_eval_classification(
    nom: str,
    modele,
    param_grid: dict,
    cached_params: dict | None,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
) -> dict:
    import warnings; warnings.filterwarnings("ignore")
    import numpy as _np
    from sklearn.base import clone as sk_clone
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, roc_curve, confusion_matrix,
    )

    best_params: dict = {}
    gs_run = False

    if cached_params:
        best_params = cached_params
    elif param_grid:
        gs = GridSearchCV(sk_clone(modele), param_grid, cv=3, scoring="roc_auc", n_jobs=1)
        gs.fit(X, y)
        best_params = gs.best_params_
        gs_run = True

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    accs, f1s, precs, recs, aucs = [], [], [], [], []

    # Pour la courbe ROC moyenne et la matrice de confusion agrégée
    base_fpr = _np.linspace(0, 1, 100)
    tprs = []
    all_y_true, all_y_pred = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        m = sk_clone(modele)
        if best_params:
            m.set_params(**best_params)
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)

        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, zero_division=0))
        precs.append(precision_score(y_te, y_pred, zero_division=0))
        recs.append(recall_score(y_te, y_pred, zero_division=0))

        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(y_pred.tolist())

        # Score pour AUC
        if hasattr(m, "predict_proba"):
            y_score = m.predict_proba(X_te)[:, 1]
        elif hasattr(m, "decision_function"):
            y_score = m.decision_function(X_te)
        else:
            y_score = y_pred.astype(float)

        try:
            aucs.append(roc_auc_score(y_te, y_score))
            fpr, tpr, _ = roc_curve(y_te, y_score)
            tprs.append(_np.interp(base_fpr, fpr, tpr))
        except Exception:
            pass

    # Matrice de confusion agrégée sur tous les folds
    cm = confusion_matrix(all_y_true, all_y_pred).tolist()
    mean_tpr = _np.mean(tprs, axis=0).tolist() if tprs else base_fpr.tolist()

    return {
        "Modèle":           nom,
        "Accuracy":         float(_np.mean(accs)),
        "Accuracy ±":       float(_np.std(accs)),
        "F1":               float(_np.mean(f1s)),
        "F1 ±":             float(_np.std(f1s)),
        "Précision":        float(_np.mean(precs)),
        "Précision ±":      float(_np.std(precs)),
        "Rappel":           float(_np.mean(recs)),
        "Rappel ±":         float(_np.std(recs)),
        "AUC":              float(_np.mean(aucs)) if aucs else float("nan"),
        "AUC ±":            float(_np.std(aucs))  if aucs else float("nan"),
        "Meilleurs params": str(best_params) if best_params else "—",
        "_params_raw":      best_params,
        "_gs_run":          gs_run,
        "_confusion_matrix": cm,
        "_fpr":             base_fpr.tolist(),
        "_tpr":             mean_tpr,
    }

```

```
async def evaluer_classification(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict, dict]:
    """
    Retourne (DataFrame résultats, dict {nom_modèle: (_fpr, _tpr)})
    pour les courbes ROC.
    """
    cols = [c for c in tl.FEATURES if c in df.columns]
    X    = df[cols].values.astype(float)
    y    = (df[tl.TARGET] >= tl.SEUIL_RETARD).to_numpy(dtype=int)

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    print(f"  Classe 0 (à l'heure) : {n_neg:,}  |  Classe 1 (retard >{tl.SEUIL_RETARD}s) : {n_pos:,}")
    print(f"  Évaluation : StratifiedKFold k={tl.N_FOLDS}  ({len(X):,} lignes)")
    print(f"  Lancement de {len(MODELES_CLASSIFICATION)} modèles en parallèle…\n")

    cache = _lire_cache(_CACHE_CLASSIFICATION)
    loop  = asyncio.get_event_loop()

    SVC_MAX_SAMPLES = 10_000
    if len(X) > SVC_MAX_SAMPLES:
        rng = np.random.default_rng(42)
        svc_idx = rng.choice(len(X), size=SVC_MAX_SAMPLES, replace=False)
        X_svc, y_svc = X[svc_idx], y[svc_idx]
        print(f"  SVC limité à {SVC_MAX_SAMPLES:,} lignes tirées au hasard (dataset trop grand)")
    else:
        X_svc, y_svc = X, y

    executor = ProcessPoolExecutor()
    futures = {
        loop.run_in_executor(
            executor,
            _train_eval_classification,
            nom, modele,
            tl.PARAM_GRIDS_CLASSIFICATION.get(nom, {}),
            cache.get(nom),
            X_svc if nom == "SVC" else X,
            y_svc if nom == "SVC" else y,
            tl.N_FOLDS,
        ): nom
        for nom, modele in MODELES_CLASSIFICATION.items()
    }

    lignes: list[dict] = []
    roc_data: dict[str, tuple] = {}
    cm_data:  dict[str, list]  = {}

    for future in asyncio.as_completed(futures):
        m = await future
        lignes.append(m)

        if m["_gs_run"] and m["_params_raw"]:
            _maj_cache(_CACHE_CLASSIFICATION, m["Modèle"], m["_params_raw"])
            print(f"  [GridSearch OK] {m['Modèle']:<25} → {m['_params_raw']}")
        elif m["_params_raw"]:
            print(f"  [Cache utilisé] {m['Modèle']:<25} → {m['_params_raw']}")

        print(
            f"  [Terminé ✓]     {m['Modèle']:<25}"
            f"  AUC={m['AUC']:.3f}±{m['AUC ±']:.3f}"
            f"  F1={m['F1']:.3f}±{m['F1 ±']:.3f}"
            f"  Acc={m['Accuracy']:.3f}±{m['Accuracy ±']:.3f}\n"
        )

        # Afficher la matrice de confusion dans le terminal
        cm = m["_confusion_matrix"]
        if len(cm) == 2:
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            print(f"    Matrice de confusion (agrégée) :")
            print(f"      TN={tn:,}  FP={fp:,}")
            print(f"      FN={fn:,}  TP={tp:,}\n")

        roc_data[m["Modèle"]] = (m["_fpr"], m["_tpr"])
        cm_data[m["Modèle"]]  = m["_confusion_matrix"]

    executor.shutdown(wait=False)

    for m in lignes:
        m.pop("_params_raw",      None)
        m.pop("_gs_run",          None)
        m.pop("_confusion_matrix",None)
        m.pop("_fpr",             None)
        m.pop("_tpr",             None)

    return pd.DataFrame(lignes), roc_data, cm_data
```

```
# ══════════════════════════════════════════
# CLASSIFICATION
# ══════════════════════════════════════════
    print("\n" + "=" * 65)
    print(f"  CLASSIFICATION  (cible : retard > {tl.SEUIL_RETARD}s)")
    print("=" * 65)

    res_clf, roc_data, cm_data = await evaluer_classification(df, "ML-ready")

    _afficher_classements_classification(res_clf)

    out_csv_clf = os.path.join(_DOSSIER, f"resultats_classification_{horodatage}.csv")
    res_clf.to_csv(out_csv_clf, index=False)
    print(f"\nRésultats CSV classification → {out_csv_clf}")

    out_roc = tracer_courbes_roc(roc_data, horodatage)
    print(f"Courbes ROC (PNG)            → {out_roc}")

    out_cms = tracer_matrices_confusion(cm_data, horodatage)
    for p in out_cms:
        print(f"Matrice de confusion (PNG)   → {p}")

    out_md_clf = ecrire_rapport_classification(
        resultats    = res_clf,
        dossier      = _DOSSIER,
        horodatage   = horodatage,
        csv_source   = args.csv,
        n_lignes     = len(df),
        features     = tl.FEATURES,
        n_folds      = tl.N_FOLDS,
        seuil_retard = tl.SEUIL_RETARD,
        roc_png      = os.path.basename(out_roc),
        cm_pngs      = [os.path.basename(p) for p in out_cms],
    )
    print(f"Rapport Markdown classification → {out_md_clf}")
```