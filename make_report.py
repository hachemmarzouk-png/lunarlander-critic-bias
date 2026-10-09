"""Generate the four-page A4 report (French) from the CSVs in results/ and figures/.

Real numbers are shown only if results exist; without them, the report is a protocol
document with empty result tables. Re-run after training (train_all.py does it).
The automatic text only states what the numbers say; the discussion should be
rewritten by the author once the results are known.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

# DejaVu is shipped with matplotlib, so Greek letters and math symbols render everywhere.
FONT_DIR = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
for name, file in [('DejaVu', 'DejaVuSans.ttf'), ('DejaVuBold', 'DejaVuSans-Bold.ttf'),
                   ('DejaVuIt', 'DejaVuSans-Oblique.ttf'),
                   ('DejaVuBoldIt', 'DejaVuSans-BoldOblique.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
addMapping('DejaVu', 0, 0, 'DejaVu')
addMapping('DejaVu', 1, 0, 'DejaVuBold')
addMapping('DejaVu', 0, 1, 'DejaVuIt')
addMapping('DejaVu', 1, 1, 'DejaVuBoldIt')
NORMAL, BOLD = 'DejaVu', 'DejaVuBold'

PAGE_W, PAGE_H = A4
NAVY, BLUE, GREY, LIGHT = (HexColor('#172B45'), HexColor('#2563A4'),
                           HexColor('#475569'), HexColor('#E2E8F0'))
M = 46
CONTENT = PAGE_W - 2 * M
BOTTOM = 46
CONFIGS = [('ddpg', 'no', 'DDPG'), ('ddpg', 'yes', 'DDPG + LN'),
           ('td3', 'no', 'TD3'), ('td3', 'yes', 'TD3 + LN')]


class Paper:
    def __init__(self, c):
        self.c = c
        self.y = PAGE_H - 50
        self.page = 0
        self.body = ParagraphStyle('Body', fontName=NORMAL, fontSize=9.3, leading=13.2,
                                   textColor=NAVY, alignment=4)
        self.small = ParagraphStyle('Small', parent=self.body, fontSize=8.2, leading=11.4)

    def new(self, subtitle):
        if self.page > 0:
            self.c.showPage()
        self.page += 1
        self.y = PAGE_H - 46
        self.c.setFont(BOLD, 8)
        self.c.setFillColor(BLUE)
        self.c.drawString(M, self.y, 'LUNARLANDER-v3  ·  BIAIS DES CRITICS')
        self.c.setFont(NORMAL, 8)
        self.c.drawRightString(PAGE_W - M, self.y, subtitle)
        self.y -= 9
        self.c.setStrokeColor(LIGHT)
        self.c.line(M, self.y, PAGE_W - M, self.y)
        self.y -= 18

    def check(self, h, what):
        if self.y - h < BOTTOM:
            raise RuntimeError(f'Page {self.page} overflow at {what!r} (y={self.y:.0f}, h={h:.0f})')

    def label(self, s):
        self.check(30, s)
        self.y -= 5
        self.c.setFont(BOLD, 10.5)
        self.c.setFillColor(NAVY)
        self.c.drawString(M, self.y - 10, s)
        self.y -= 17

    def para(self, content, small=False, space=7):
        p = Paragraph(content, self.small if small else self.body)
        _, h = p.wrap(CONTENT, 2000)
        self.check(h, content[:40])
        p.drawOn(self.c, M, self.y - h)
        self.y -= h + space

    def eq(self, text):
        h = 22
        self.check(h, text)
        self.c.setFillColor(HexColor('#F1F5F9'))
        self.c.roundRect(M, self.y - h, CONTENT, h, 5, stroke=0, fill=1)
        self.c.setFont(NORMAL, 9)
        self.c.setFillColor(NAVY)
        self.c.drawCentredString(PAGE_W / 2, self.y - 14.5, text)
        self.y -= h + 7

    def table(self, headers, rows, widths, fontsize=7.8, rowh=17):
        """Cells may contain '\\n': the second line is drawn smaller and grey (CIs)."""
        two_lines = any('\n' in str(v) for r in rows for v in r)
        body_h = rowh + (9 if two_lines else 0)
        total = rowh + body_h * len(rows)
        self.check(total, 'table')
        y = self.y
        self.c.setFillColor(NAVY)
        self.c.rect(M, y - rowh, CONTENT, rowh, fill=1, stroke=0)
        x = M
        self.c.setFont(BOLD, fontsize)
        self.c.setFillColor(HexColor('#FFFFFF'))
        for w, h in zip(widths, headers):
            self.c.drawString(x + 6, y - rowh + 5.5, str(h))
            x += w
        y -= rowh
        for ri, row in enumerate(rows):
            if ri % 2 == 1:
                self.c.setFillColor(HexColor('#F3F7FC'))
                self.c.rect(M, y - body_h, CONTENT, body_h, fill=1, stroke=0)
            x = M
            for w, value in zip(widths, row):
                first, _, second = str(value).partition('\n')
                self.c.setFont(NORMAL, fontsize)
                self.c.setFillColor(NAVY)
                self.c.drawString(x + 6, y - 12, first)
                if second:
                    self.c.setFont(NORMAL, fontsize - 1.3)
                    self.c.setFillColor(GREY)
                    self.c.drawString(x + 6, y - 21.5, second)
                x += w
            y -= body_h
        self.c.setStrokeColor(LIGHT)
        self.c.line(M, y, PAGE_W - M, y)
        self.y = y - 9

    def img(self, path, height):
        self.check(height, str(path))
        self.c.drawImage(str(path), M, self.y - height, width=CONTENT, height=height,
                         preserveAspectRatio=True, anchor='c', mask='auto')
        self.y -= height + 3

    def foot(self, tag):
        self.c.setStrokeColor(LIGHT)
        self.c.line(M, 34, PAGE_W - M, 34)
        self.c.setFont(NORMAL, 7)
        self.c.setFillColor(GREY)
        self.c.drawString(M, 23, tag)
        self.c.drawRightString(PAGE_W - M, 23, f'{self.page} / 4')


def fmt(v, d=1, sign=False):
    if v is None or pd.isna(v):
        return 'ND'
    s = f'{v:+.{d}f}' if sign else f'{v:.{d}f}'
    return s.replace('-', '−').replace('.', ',')


def cell(row, m, d=1, sign=False):
    if row is None or m not in row or pd.isna(row[m]):
        return 'ND'
    ci = ''
    if pd.notna(row.get(f'{m}_lo')):
        ci = f'\n[{fmt(row[m + "_lo"], d, sign)} ; {fmt(row[m + "_hi"], d, sign)}]'
    return fmt(row[m], d, sign) + ci


def load(results: Path, figs: Path):
    if not list(results.glob('*/evaluations.csv')):
        return None, None, None, None
    df = pd.concat((pd.read_csv(p) for p in sorted(results.glob('*/evaluations.csv'))),
                   ignore_index=True)
    df['layer_norm'] = df['layer_norm'].astype(str)
    summary = effects = final = None
    if (figs / 'final_summary.csv').exists():
        summary = pd.read_csv(figs / 'final_summary.csv')
        summary['layer_norm'] = summary['layer_norm'].astype(str)
    if (figs / 'effects.csv').exists():
        effects = pd.read_csv(figs / 'effects.csv')
    if (figs / 'final_by_seed.csv').exists():
        final = pd.read_csv(figs / 'final_by_seed.csv')
    return df, summary, effects, final


def effect_sentence(effects, metric, contrast, unit, d=1):
    if effects is None or effects.empty:
        return None
    e = effects[(effects.metric == metric) & (effects.contrast == contrast)]
    if e.empty:
        return None
    e = e.iloc[0]
    verdict = ('l’intervalle exclut 0' if e.ci_excludes_0
               else 'l’intervalle contient 0 : effet non établi avec ces graines')
    return (f'{fmt(e.estimate, d, True)}{unit} (IC 95 % [{fmt(e.ci_lo, d, True)} ; '
            f'{fmt(e.ci_hi, d, True)}], {verdict})')


def sup(n):
    return str(n).translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))


def nb(n):
    return f'{int(n):,}'.replace(',', ' ')


def protocol_meta(results: Path, df):
    """Protocol numbers shown in the text: from the runs if present, else the defaults."""
    import json
    m = dict(seeds=5, runs=20, steps=300_000, eval_every=25_000, eval_episodes=20, tail=500)
    configs = sorted(results.glob('*/config.json'))
    if configs:
        c = json.loads(configs[0].read_text(encoding='utf8'))
        m.update(steps=c.get('steps', m['steps']), eval_every=c.get('eval_every', m['eval_every']),
                 eval_episodes=c.get('eval_episodes', m['eval_episodes']),
                 tail=c.get('eval_tail', m['tail']))
    if df is not None:
        m['seeds'] = int(df.seed.nunique())
        m['runs'] = int(df.groupby(['algorithm', 'layer_norm', 'seed']).ngroups)
    return m


def build(path: Path, results: Path, figs: Path, author: str = 'Hachem'):
    df, summary, effects, final = load(results, figs)
    meta = protocol_meta(results, df)
    has = df is not None and summary is not None
    c = canvas.Canvas(str(path), pagesize=A4, pageCompression=1)
    c.setTitle('Biais des critics en Deep RL : DDPG, TD3 et Layer Normalization')
    c.setAuthor(author)
    p = Paper(c)

    # ---------------------------------------------------------------- PAGE 1
    p.new('question et cadre')
    c.setFillColor(NAVY)
    c.setFont(BOLD, 16.5)
    c.drawString(M, p.y - 4, 'Biais d’estimation des critics en Deep RL')
    p.y -= 23
    c.setFont(NORMAL, 10)
    c.setFillColor(GREY)
    c.drawString(M, p.y, 'DDPG vs TD3, avec et sans Layer Normalization — LunarLander-v3 continu')
    p.y -= 14
    c.setFont(NORMAL, 9)
    c.drawString(M, p.y, author)
    p.y -= 20
    if has:
        n_runs = len(final) if final is not None else df.groupby(
            ['algorithm', 'layer_norm', 'seed']).ngroups
        status = (f'Les résultats portent sur {n_runs} entraînements indépendants '
                  f'(jusqu’à {int(df.step.max()):,} interactions chacun).'.replace(',', ' '))
    else:
        status = ('<b>Version protocole :</b> les expériences n’ont pas encore été exécutées ; '
                  'les tableaux de résultats sont vides.')
    p.para('<b>Résumé.</b> Un actor-critic apprend sa fonction de valeur par bootstrap : '
           'ses cibles contiennent ses propres estimations, et la maximisation de l’actor '
           'contre ce critic peut transformer le bruit d’approximation en surestimation '
           'systématique. Nous mesurons ce biais pour DDPG et TD3, avec et sans LayerNorm, '
           'sur <i>LunarLander-v3</i> à actions continues, en comparant à intervalles réguliers '
           'les valeurs prédites Q(s<sub>t</sub>, a<sub>t</sub>) aux retours Monte-Carlo '
           'G<sub>t</sub> effectivement obtenus par la politique jusqu’à la fin de l’épisode. '
           'Un plan factoriel 2 × 2 répliqué sur plusieurs graines sépare l’effet de '
           'l’algorithme, celui de la normalisation et leur interaction. ' + status, space=11)
    p.label('1. Introduction')
    p.para('Dans les méthodes actor-critic, le critic Q<sub>θ</sub> approxime la valeur '
           'd’une action et l’actor est entraîné à maximiser ce critic. Thrun et Schwartz '
           '(1993) puis van Hasselt (2010) ont montré que maximiser une estimation bruitée '
           'biaise vers le haut : si ε est un bruit centré, E[maxₐ (Q(a) + ε(a))] ≥ maxₐ Q(a). Fujimoto et al. [1] observent ce '
           'phénomène dans DDPG [2] et proposent TD3, dont la cible prend le minimum de deux '
           'critics (<i>clipped double Q-learning</i>), lisse l’action cible et retarde les '
           'mises à jour de l’actor. La LayerNorm [3], qui renormalise les activations '
           'cachées, est par ailleurs de plus en plus utilisée dans les critics pour '
           'stabiliser l’apprentissage et limiter l’extrapolation des valeurs.')
    p.para('Nous posons deux questions : <b>(Q1)</b> quel biais observe-t-on réellement '
           'pour chaque méthode au cours de l’entraînement, et TD3 le réduit-il par rapport à '
           'DDPG ? <b>(Q2)</b> la LayerNorm modifie-t-elle le signe, l’amplitude ou la '
           'stabilité de ce biais, et la performance de la politique ?')
    p.label('2. Formulation')
    p.para('Pour une politique déterministe π et un facteur d’actualisation γ, la valeur '
           'd’action est le retour actualisé espéré après avoir joué a en s puis suivi π. '
           'Elle vérifie l’équation de Bellman :', space=5)
    p.eq('Qπ(s, a) = E[ r + γ Qπ(s′, π(s′)) | s, a ],      Qπ(s, a) = E[ Σk≥0 γᵏ rₜ₊ₖ | sₜ = s, aₜ = a ]')
    p.para('Les deux algorithmes régressent le critic vers une cible construite avec des '
           'copies lentes (moyenne de Polyak) de l’actor et du ou des critics :', space=5)
    p.eq('y_DDPG = r + γ (1 − d) Q′(s′, π′(s′))')
    p.eq('y_TD3 = r + γ (1 − d) min{ Q′₁(s′, ã), Q′₂(s′, ã) },   ã = clip(π′(s′) + clip(ε, −c, c))')
    p.para('où d = 1 seulement pour une vraie terminaison (atterrissage ou crash) : la '
           'coupure à 1000 pas n’est pas un état absorbant et l’on continue d’y '
           'bootstrapper. Le critic estime donc la valeur à horizon infini, sans notion de '
           'temps restant. L’actor maximise Q₁(s, π(s)). On définit le <b>biais</b> au point '
           '(s, a) comme Q<sub>θ</sub>(s, a) − Qπ(s, a), que l’on estime en moyenne sur les '
           'états visités par π.', small=True)
    p.foot('Biais des critics · DDPG / TD3 / LayerNorm')

    # ---------------------------------------------------------------- PAGE 2
    p.new('protocole')
    p.label('3. Protocole expérimental')
    p.para('<b>Environnement.</b> <i>Gymnasium LunarLander-v3</i> avec <b>continuous=True</b> : '
           'observation de dimension 8, action (moteur principal, moteurs latéraux) dans '
           '[−1, 1]², limite de 1000 pas [4]. Un crash coûte −100, un atterrissage réussi '
           'rapporte +100 ; la tâche est considérée résolue au-delà de 200 points. Le '
           'moteur a une dispersion aléatoire : la dynamique est stochastique.')
    p.para('<b>Plan factoriel.</b> Facteur algorithme (DDPG, TD3) × facteur normalisation '
           f'(sans, avec LayerNorm), {meta["seeds"]} graines indépendantes par cellule, soit '
           f'{meta["runs"]} entraînements de {nb(meta["steps"])} interactions. Tous les autres '
           'réglages sont identiques.')
    p.table(['Hyperparamètre', 'Valeur', 'Hyperparamètre', 'Valeur'],
            [['Réseaux actor / critic', 'MLP 256-256, ReLU', 'Sortie actor', 'tanh'],
             ['γ  /  τ (Polyak)', '0,99  /  0,005', 'Pas Adam actor / critic', '10⁻³ / 10⁻³'],
             ['Mini-batch  /  replay', '256  /  300 000', 'Exploration', 'gaussienne σ = 0,1'],
             ['Pas aléatoires initiaux', '10 000', 'TD3 : bruit cible', 'σ = 0,2, clip 0,5'],
             ['LayerNorm', 'couches cachées, avant ReLU', 'TD3 : délai actor', '2']],
            widths=[118, 150, 118, CONTENT - 386])
    p.para('<b>Variantes.</b> Notre DDPG est la version ré-ajustée de [1] (« OurDDPG ») : '
           'même architecture, mêmes pas d’apprentissage et même exploration gaussienne que '
           'TD3, sans bruit d’Ornstein-Uhlenbeck ni pénalité L2. Ce choix isole les '
           'mécanismes propres à TD3. La LayerNorm est appliquée aux couches cachées de '
           'l’actor et des critics (et de leurs copies cibles), jamais aux sorties.', small=True)
    p.label('4. Mesure Monte-Carlo du biais')
    p.para(f'Toutes les {nb(meta["eval_every"])} interactions, l’actor est figé et joué '
           f'<b>sans bruit</b> sur {meta["eval_episodes"]} '
           'épisodes d’évaluation dont les conditions initiales sont identiques pour tous '
           'les entraînements et tous les checkpoints. Pour chaque état-action visité, on '
           'calcule le retour actualisé réellement obtenu jusqu’à la fin de l’épisode :', space=5)
    p.eq('Gₜ = rₜ + γ rₜ₊₁ + γ² rₜ₊₂ + … (jusqu’à la fin),     biais = E[ Q₁(sₜ, aₜ) − Gₜ ],     MAE = E| Q₁(sₜ, aₜ) − Gₜ |')
    p.para('Gₜ est un échantillon sans biais de Qπ(sₜ, aₜ) qui ne dépend d’aucun réseau. Les '
           'écarts sont moyennés dans chaque épisode puis entre épisodes (l’épisode, pas le '
           'pas de temps, est l’unité d’échantillonnage), puis comparés entre graines. On '
           'rapporte aussi un <b>biais relatif</b> E[Q₁ − G] / E|G|, sans unité, car des '
           'politiques de niveaux différents ont des retours d’échelles différentes '
           '(normaliser par |E[G]| comme dans REDQ [5] est instable ici, E[G] traversant 0 '
           'entre crash et atterrissage). Pour TD3, Q₁ sert de référence (même rôle que le '
           'critic unique de DDPG) ; Q₂ et min(Q₁, Q₂) sont aussi enregistrés.')
    tail = meta['tail']
    frac = f'{100 * 0.99 ** tail:.1f}'.replace('.', ',')
    p.para('<b>Coupure à 1000 pas.</b> Une politique qui plane sans se poser est coupée par '
           'la limite de temps ; un retour Monte-Carlo arrêté là omettrait la queue de la '
           'somme, et exclure ces épisodes écarterait précisément les politiques où la '
           'surestimation est la plus probable (biais de sélection). L’épisode d’évaluation '
           f'est donc <b>prolongé de H = {tail} pas</b> : la performance reste la somme des '
           '1000 premières récompenses, mais Gₜ est calculé sur le déroulé prolongé pour tout '
           f't < 1000. L’erreur de troncature résiduelle est au plus γ{sup(tail)}·max|r| / (1 − γ), '
           f'soit {frac} % de l’échelle maximale des valeurs. Aucun épisode n’est exclu.',
           small=True)
    p.foot('Implémentation : PyTorch + Gymnasium · protocole fixé avant les expériences')

    # ---------------------------------------------------------------- PAGE 3
    p.new('résultats')
    p.label('5. Résultats')
    rows = []
    for alg, ln, label in CONFIGS:
        r = None
        if has:
            part = summary[(summary.algorithm == alg) & (summary.layer_norm == ln)]
            r = part.iloc[0].to_dict() if not part.empty else None
        rows.append([label + (f'\n{int(r["n_seeds"])} graines' if r else ''),
                     cell(r, 'mean_return', 0), cell(r, 'q1_bias', 1, True),
                     cell(r, 'q1_rel_bias', 2, True), cell(r, 'q1_mae', 1),
                     cell(r, 'timeout_episodes', 1)])
    if has:
        lastk = ''
        if final is not None and not final.empty:
            lastk = (f' entre {int(final.first_step.min()):,} et '
                     f'{int(final.last_step.max()):,} interactions').replace(',', ' ')
        p.para('Valeurs finales : moyenne des trois derniers checkpoints de chaque '
               f'entraînement{lastk}, puis moyenne entre graines ; entre crochets, intervalle '
               'de confiance à 95 % par bootstrap sur les graines (10 000 rééchantillonnages). '
               f'Avec {meta["seeds"]} graines par cellule, ces intervalles restent approximatifs.',
               small=True, space=6)
    else:
        p.para('<b>Expériences non exécutées.</b> Le tableau est volontairement vide ; '
               '<font name="DejaVu" color="#2563A4">python train_all.py</font> exécute la '
               'grille et régénère ce rapport avec les mesures.', small=True, space=6)
    p.table(['Configuration', 'Retour', 'Biais Q₁', 'Biais relatif', 'MAE Q₁', 'En vol à 1000'],
            rows, widths=[96, 82, 82, 86, 76, CONTENT - 422])
    if has:
        for picture, h, caption in [
                ('bias.png', 196, 'Figure 1 — Biais signé E[Q₁ − G] au cours de l’entraînement '
                 '(> 0 : surestimation). Bandes : IC 95 % bootstrap sur les graines.'),
                ('return.png', 196, 'Figure 2 — Retour d’évaluation de la politique '
                 'déterministe (somme des 1000 premières récompenses).')]:
            if (figs / picture).is_file():
                p.img(figs / picture, h)
                p.para(caption, small=True, space=6)
    else:
        p.para('<b>Hypothèses.</b> <b>H1</b> — DDPG surestime (biais > 0) et le minimum des '
               'deux critics de TD3 réduit ce biais, au risque de le rendre négatif '
               '(sous-estimation) [1]. <b>H2</b> — La LayerNorm borne l’amplitude des '
               'activations et peut limiter les valeurs extrêmes du critic ; son effet sur le '
               'biais peut différer entre DDPG et TD3 (interaction). <b>H3</b> — Un critic '
               'peu biaisé en moyenne n’est pas forcément précis (erreurs de signes opposés '
               'qui se compensent) : biais signé et MAE doivent être lus ensemble, et '
               'séparément de la performance.')
    p.foot('ND = non disponible · retour en points LunarLander · « En vol à 1000 » : épisodes '
           f'd’évaluation (sur {meta["eval_episodes"]}) non terminés à la limite')

    # ---------------------------------------------------------------- PAGE 4
    p.new('effets et discussion')
    p.label('6. Effets du plan factoriel')
    eff_rows = []
    for contrast in ['TD3 - DDPG (moyenne sur LN)', 'LN - sans LN (moyenne sur algo)',
                     'Interaction algo x LN', 'LN - sans LN | DDPG', 'LN - sans LN | TD3']:
        row = [contrast.replace(' - ', ' − ').replace(' x ', ' × ')]
        for metric, d in [('q1_bias', 1), ('q1_rel_bias', 2), ('q1_mae', 1), ('mean_return', 0)]:
            e = None
            if has and effects is not None and not effects.empty:
                sel = effects[(effects.metric == metric) & (effects.contrast == contrast)]
                e = sel.iloc[0] if not sel.empty else None
            if e is None:
                row.append('ND')
            else:
                star = ' *' if e.ci_excludes_0 else ''
                row.append(f'{fmt(e.estimate, d, True)}{star}\n'
                           f'[{fmt(e.ci_lo, d, True)} ; {fmt(e.ci_hi, d, True)}]')
        eff_rows.append(row)
    p.table(['Contraste', 'Biais Q₁', 'Biais relatif', 'MAE Q₁', 'Retour'], eff_rows,
            widths=[163, 85, 85, 85, CONTENT - 418])
    p.para('Différences de moyennes entre cellules, IC 95 % par bootstrap des graines dans '
           'chaque cellule ; * : l’intervalle exclut 0. Interaction = (TD3+LN − TD3) − '
           '(DDPG+LN − DDPG).', small=True, space=8)
    p.label('7. Discussion')
    s_alg = effect_sentence(effects, 'q1_bias', 'TD3 - DDPG (moyenne sur LN)', ' points') if has else None
    s_ln = effect_sentence(effects, 'q1_bias', 'LN - sans LN (moyenne sur algo)', ' points') if has else None
    if s_alg and s_ln:
        p.para(f'<b>Constats chiffrés.</b> Passer de DDPG à TD3 change le biais signé de Q₁ de '
               f'{s_alg}. Ajouter la LayerNorm le change de {s_ln}. '
               '<i>[Interprétation à rédiger : signe du biais de chaque méthode, évolution au '
               'cours de l’entraînement (figure 1), lien avec la performance (figure 2).]</i>')
    p.para('<b>Lecture.</b> Un biais positif signifie que le critic surestime, <i>en moyenne '
           'sur les états-actions que la politique visite</i>, et non pour toutes les '
           'actions. Une réduction du biais signé par TD3 serait cohérente avec la '
           'motivation du minimum de deux critics [1] ; un biais devenu négatif traduirait '
           'la sous-estimation que ce minimum peut induire. La MAE indique si la réduction '
           'du biais vient d’une meilleure précision ou d’erreurs qui se compensent.')
    p.label('8. Limites')
    p.para('<b>(i)</b> Gₜ est un échantillon bruité : les conclusions reposent sur la '
           f'réplication (épisodes, graines), et {meta["seeds"]} graines par cellule donnent des intervalles larges. '
           '<b>(ii)</b> Chaque politique visite ses propres états : le biais est pondéré par '
           'l’occupation de chaque politique, pas mesuré à états fixés. <b>(iii)</b> TD3 diffère '
           'de DDPG par trois mécanismes (double critic, lissage de la cible, délai) que ce '
           'plan ne sépare pas ; de plus, le lissage fait apprendre au critic la valeur d’une '
           'politique légèrement bruitée, alors que Gₜ est mesuré sans bruit. <b>(iv)</b> La '
           'LayerNorm agit à la fois sur l’actor et sur les critics : son effet sur le seul '
           'critic n’est pas isolé. <b>(v)</b> Un seul environnement, un seul budget, un seul '
           'jeu d’hyperparamètres.', small=True, space=8)
    p.label('Références')
    p.para('[1] S. Fujimoto, H. van Hoof, D. Meger. Addressing Function Approximation Error '
           'in Actor-Critic Methods. <i>ICML</i>, 2018. — '
           '[2] T. Lillicrap et al. Continuous Control with Deep Reinforcement Learning. '
           '<i>ICLR</i>, 2016. — '
           '[3] J. L. Ba, J. R. Kiros, G. Hinton. Layer Normalization. arXiv:1607.06450, 2016. — '
           '[4] Farama Foundation. Gymnasium, <i>Lunar Lander</i>. — '
           '[5] X. Chen, C. Wang, Z. Zhou, K. Ross. Randomized Ensembled Double Q-Learning '
           '(REDQ). <i>ICLR</i>, 2021. — '
           'S. Thrun, A. Schwartz. Issues in Using Function Approximation for Reinforcement '
           'Learning, 1993. — H. van Hasselt. Double Q-learning. <i>NeurIPS</i>, 2010.',
           small=True, space=0)
    p.foot('Code, données et figures : dépôt du projet')
    c.save()
    print('Generated', path, f'- {p.page} pages; checkpoints:', 0 if df is None else len(df))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results')
    parser.add_argument('--figures', default='figures')
    parser.add_argument('--output', default='rapport_4_pages.pdf')
    parser.add_argument('--author', default='Hachem')
    args = parser.parse_args()
    build(Path(args.output), Path(args.results), Path(args.figures), args.author)


if __name__ == '__main__':
    main()
