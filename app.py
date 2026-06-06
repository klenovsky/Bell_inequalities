
import io
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Bell Inequalities Explorer", layout="wide")

# ---------------------------
# Text content
# ---------------------------

TEXT = {
    "en": {
        "title": "Bell Inequalities Explorer",
        "subtitle": "An interactive view of entanglement, Bell correlations, CHSH violation, and the boundary between quantum theory and local hidden-variable models.",
        "language": "Language",
        "theory_title": "Theory, context, and references",
        "theory_body": r"""
This app sits at the intersection of **quantum mechanics**, **entanglement**, and the question of whether the correlations predicted by quantum theory can be reproduced by any **local hidden-variable model**. The historical background begins with the EPR argument, continues with Bell’s theorem, and becomes experimentally concrete in the CHSH form used in many laboratory tests [1–5].

### Physical idea

For two spin-$1/2$ particles in the singlet state

$$
|\psi^{-}\rangle = \frac{|01\rangle - |10\rangle}{\sqrt{2}},
$$

the measurement outcomes at two distant analyzers are individually random, but their **joint correlations** are highly structured. If Alice measures along axis $a$ and Bob along axis $b$, quantum mechanics predicts

$$
E(a,b) = \langle \sigma_a \otimes \sigma_b \rangle = -\cos(a-b)
$$

for the ideal singlet in a common measurement plane. That cosine law is stronger than what is possible in the simplest local realistic descriptions and leads to violations of Bell inequalities [2–4].

### CHSH form

A common Bell parameter is

$$
S = E(a,b) + E(a,b') + E(a',b) - E(a',b').
$$

Any local hidden-variable theory of the CHSH type satisfies

$$
|S| \le 2,
$$

whereas quantum mechanics can reach the **Tsirelson bound**

$$
|S| \le 2\sqrt{2}.
$$

The usual maximizing choice is a relative analyzer geometry of $45^\circ$ between the relevant settings [2,3,6].

### Why this matters

Bell tests do not merely show that quantum mechanics has unusual correlations. They show that the conjunction of **locality**, **realism**, and certain probabilistic assumptions cannot reproduce all quantum predictions. In modern language, Bell nonlocality sits above entanglement in the hierarchy of nonclassical correlations: all Bell-nonlocal states are entangled, but not all entangled states violate a Bell inequality under a given test [4,6–8].

### Minimal modelling choice used here

This simulator keeps the geometry deliberately simple:
- two qubits,
- analyzer axes restricted to one plane,
- the singlet state as the main reference state,
- a simple deterministic local hidden-variable comparison model,
- a noisy **Werner-state visibility** panel,
- event-by-event Monte Carlo accumulation for a CHSH experiment.

That makes the structure of the correlations easy to inspect, while preserving the core logic of Bell’s theorem and the CHSH test [2–8].

### References

[1] A. Einstein, B. Podolsky, and N. Rosen, *Can Quantum-Mechanical Description of Physical Reality Be Considered Complete?* Phys. Rev. **47**, 777 (1935).  
[2] J. S. Bell, *On the Einstein Podolsky Rosen paradox*, Physics Physique Fizika **1**, 195 (1964).  
[3] J. F. Clauser, M. A. Horne, A. Shimony, and R. A. Holt, *Proposed Experiment to Test Local Hidden-Variable Theories*, Phys. Rev. Lett. **23**, 880 (1969).  
[4] N. Brunner, D. Cavalcanti, S. Pironio, V. Scarani, and S. Wehner, *Bell nonlocality*, Rev. Mod. Phys. **86**, 419 (2014).  
[5] A. Aspect, J. Dalibard, and G. Roger, *Experimental Test of Bell's Inequalities Using Time-Varying Analyzers*, Phys. Rev. Lett. **49**, 1804 (1982).  
[6] B. S. Cirel'son, *Quantum generalizations of Bell's inequality*, Lett. Math. Phys. **4**, 93 (1980).  
[7] R. F. Werner, *Quantum states with Einstein–Podolsky–Rosen correlations admitting a hidden-variable model*, Phys. Rev. A **40**, 4277 (1989).  
[8] M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*, Cambridge University Press (2000/2010).
""",
        "global_help": """
Use the sidebar to switch language. Open the expander above each panel for a short guide to the controls and to the meaning of the graphs.  
Animated sections can be played directly in the app and can also be exported to GIF.
""",
        "panel1_title": "1. Singlet correlations: quantum vs local hidden variables",
        "panel1_help": r"""
This panel compares the ideal singlet correlation with a simple deterministic local hidden-variable model.

**Controls**
- **Angle difference** sets the current analyzer separation $\Delta = a-b$ for the snapshot.
- **Number of points** controls the resolution of the continuous curves.

**Left graph**
- Shows the correlation function $E(\Delta)$ over the full interval $0 \le \Delta \le \pi$.
- The **quantum curve** is the cosine law $-\cos\Delta$.
- The **local hidden-variable curve** is the piecewise linear correlation produced by a simple sign-threshold model. It reproduces perfect anticorrelation at $\Delta=0$, but not the full quantum cosine.

**Right graph**
- Shows the four joint probabilities $P(++), P(+-), P(-+), P(--)$ for the chosen angle difference.
- For the singlet, same-outcome probabilities are suppressed when the analyzers are aligned, while opposite outcomes dominate.
- As the angle changes, the probability weight redistributes continuously.

**How to read it**
- The key message is not that one curve is just “larger” than the other.
- The key point is that the **functional form** of the quantum correlation is different enough to enable CHSH violations for suitable angle choices [2–4].
""",
        "angle_difference": "Angle difference Δ (degrees)",
        "resolution": "Number of points",
        "panel2_title": "2. CHSH explorer",
        "panel2_help": r"""
This panel evaluates the CHSH parameter for four analyzer settings $(a,a',b,b')$.

**Controls**
- Set the four analyzer angles directly.
- The **Maximal quantum preset** button loads the standard choice that reaches $|S|=2\sqrt{2}$ for the singlet.

**Top-left graph**
- Bars show the four correlators entering the CHSH combination.
- The dashed lines indicate the classical interval and the current CHSH value.

**Top-right graph**
- A 2D heat map of the quantum CHSH value as a function of $a'$ and $b'$ while $a$ and $b$ are kept fixed.
- This makes it easy to see how special the optimal region is.

**Bottom-left graph**
- The same CHSH quantity, but for the comparison local hidden-variable model.
- It stays within the classical bound.

**Bottom-right graph**
- A compact gauge of the current $|S|$ value with markers at the classical limit $2$ and the Tsirelson bound $2\sqrt{2}$.

**How to read it**
- Bell violation is not a generic property of any angle choice.
- It appears only in certain geometries of measurement settings.
- The CHSH inequality turns the difference between local and quantum correlation structures into a single test quantity [3,4,6].
""",
        "a_angle": "a (degrees)",
        "ap_angle": "a' (degrees)",
        "b_angle": "b (degrees)",
        "bp_angle": "b' (degrees)",
        "set_maximal": "Load maximal quantum preset",
        "panel3_title": "3. Event-by-event CHSH test",
        "panel3_help": r"""
This panel simulates a Bell test as an accumulation of many detected pairs.

**Controls**
- **Number of emitted pairs** sets the total Monte Carlo sample.
- **Visibility** mixes the singlet with white noise through a Werner-state parameter $V$.
- **Animation speed** sets the playback rate.

**Animated graph**
- The solid curve shows the running estimate of the CHSH value for the simulated quantum source.
- The dotted curve shows the same quantity for a simple local hidden-variable comparison model.
- The pale dashed horizontal lines mark the classical bound $2$ and the ideal singlet Tsirelson value $2\sqrt{2}$.

**Side graph**
- Shows the running conditional correlations for the four setting pairs.
- Early points fluctuate strongly because only a few events have reached each channel.
- Later, the curves stabilize near the target values.

**What the animation means**
- Each event is local and discrete, but the Bell parameter is built from **statistics accumulated across many events**.
- The violation does not appear in a single pair; it emerges in the aggregate correlations [3–5].

**GIF export**
- The GIF export reproduces the running CHSH build-up using matplotlib and Pillow, so it does not depend on Chrome or Kaleido.
""",
        "num_pairs": "Number of emitted pairs",
        "visibility": "Visibility V",
        "animation_speed": "Animation speed (ms per frame)",
        "panel4_title": "4. Noise, visibility, entanglement, and Bell violation",
        "panel4_help": r"""
This panel shows how adding white noise changes the relation between entanglement and Bell violation.

**Controls**
- **Visibility** sets the Werner-state mixture
  $$
  \rho_W = V |\psi^{-}\rangle\langle\psi^{-}| + (1-V)\frac{\mathbb{I}}{4}.
  $$

**Left graph**
- Shows the noisy correlation function $E(\Delta)=-V\cos\Delta$ together with the classical comparison curve.
- Increasing noise shrinks the correlation amplitude.

**Right graph**
- Shows the maximal quantum CHSH value for a Werner state,
  $$
  S_{\max} = 2\sqrt{2}\,V.
  $$
- The horizontal markers indicate the classical CHSH limit and the Werner entanglement threshold.

**How to read it**
- There is a region in which the Werner state is still entangled but does not violate CHSH.
- This helps separate two ideas that are often conflated:
  - **entanglement**
  - **Bell-nonlocality**
- The simulator makes that distinction visible with a single noise slider [4,7,8].
""",
        "reset_panel": "Reset panel",
        "download_gif": "Download GIF",
        "prepare_gif": "Prepare GIF",
        "gif_expander": "Save this animation as GIF",
        "frame_ms": "Frame duration (ms)",
        "max_frames": "Maximum number of frames",
        "gif_ready": "GIF ready.",
        "quantum": "Quantum",
        "lhv": "Local hidden-variable model",
        "classical_bound": "Classical bound",
        "tsirelson": "Tsirelson bound",
        "werner_threshold": "Werner entangled threshold",
    },
    "cs": {
        "title": "Průzkumník Bellových nerovností",
        "subtitle": "Interaktivní pohled na provázání, Bellovy korelace, porušení CHSH a hranici mezi kvantovou teorií a lokálními modely se skrytými proměnnými.",
        "language": "Jazyk",
        "theory_title": "Teorie, kontext a reference",
        "theory_body": r"""
Tato aplikace leží na rozhraní **kvantové mechaniky**, **provázání** a otázky, zda lze korelace předpovězené kvantovou teorií reprodukovat nějakým **lokálním modelem se skrytými proměnnými**. Historické pozadí začíná argumentem EPR, pokračuje Bellovým teorémem a experimentálně se konkretizuje v CHSH formě používané v mnoha laboratorních testech [1–5].

### Fyzikální myšlenka

Pro dva spin-$1/2$ objekty v singletovém stavu

$$
|\psi^{-}\rangle = \frac{|01\rangle - |10\rangle}{\sqrt{2}},
$$

jsou výsledky měření na dvou vzdálených analyzátorech jednotlivě náhodné, ale jejich **společné korelace** mají silnou strukturu. Pokud Alice měří podél osy $a$ a Bob podél osy $b$, kvantová mechanika předpovídá

$$
E(a,b) = \langle \sigma_a \otimes \sigma_b \rangle = -\cos(a-b)
$$

pro ideální singlet v jedné měřicí rovině. Tento kosinusový zákon je silnější než to, co je možné v nejjednodušších lokálních realistických popisech, a vede k porušení Bellových nerovností [2–4].

### CHSH tvar

Běžný Bellův parametr má tvar

$$
S = E(a,b) + E(a,b') + E(a',b) - E(a',b').
$$

Každá lokální teorie skrytých proměnných typu CHSH splňuje

$$
|S| \le 2,
$$

zatímco kvantová mechanika může dosáhnout **Tsirelsonovy meze**

$$
|S| \le 2\sqrt{2}.
$$

Obvyklá volba maximalizující porušení odpovídá relativní geometrii analyzátorů s rozdíly $45^\circ$ mezi vhodnými směry [2,3,6].

### Proč je to důležité

Bellovy testy neukazují jen to, že kvantová mechanika má neobvyklé korelace. Ukazují, že spojení **lokality**, **realismu** a určitých pravděpodobnostních předpokladů nedokáže reprodukovat všechny kvantové předpovědi. V moderním jazyce leží Bellova nelokalita výše v hierarchii neklasických korelací než samotné provázání: všechny Bellovsky nelokální stavy jsou provázané, ale ne všechny provázané stavy porušují danou Bellovu nerovnost [4,6–8].

### Minimální model použitý zde

Tento simulátor záměrně zachovává jednoduchou geometrii:
- dva qubity,
- osy analyzátorů omezené na jednu rovinu,
- singletový stav jako hlavní referenční stav,
- jednoduchý deterministický lokální model se skrytými proměnnými pro srovnání,
- panel se šumem a **viditelností Wernerova stavu**,
- Monte Carlo akumulaci událostí pro CHSH experiment.

To zpřehledňuje strukturu korelací a současně zachovává základní logiku Bellova teorému a CHSH testu [2–8].

### Reference

[1] A. Einstein, B. Podolsky, a N. Rosen, *Can Quantum-Mechanical Description of Physical Reality Be Considered Complete?* Phys. Rev. **47**, 777 (1935).  
[2] J. S. Bell, *On the Einstein Podolsky Rosen paradox*, Physics Physique Fizika **1**, 195 (1964).  
[3] J. F. Clauser, M. A. Horne, A. Shimony a R. A. Holt, *Proposed Experiment to Test Local Hidden-Variable Theories*, Phys. Rev. Lett. **23**, 880 (1969).  
[4] N. Brunner, D. Cavalcanti, S. Pironio, V. Scarani a S. Wehner, *Bell nonlocality*, Rev. Mod. Phys. **86**, 419 (2014).  
[5] A. Aspect, J. Dalibard a G. Roger, *Experimental Test of Bell's Inequalities Using Time-Varying Analyzers*, Phys. Rev. Lett. **49**, 1804 (1982).  
[6] B. S. Cirel'son, *Quantum generalizations of Bell's inequality*, Lett. Math. Phys. **4**, 93 (1980).  
[7] R. F. Werner, *Quantum states with Einstein–Podolsky–Rosen correlations admitting a hidden-variable model*, Phys. Rev. A **40**, 4277 (1989).  
[8] M. A. Nielsen a I. L. Chuang, *Quantum Computation and Quantum Information*, Cambridge University Press (2000/2010).
""",
        "global_help": """
V postranním panelu lze přepnout jazyk. Nad každým panelem otevři rozbalovací nápovědu: vysvětluje, co dělají ovladače a jak číst grafy.  
Animované části lze spouštět přímo v aplikaci a také uložit jako GIF.
""",
        "panel1_title": "1. Singletové korelace: kvantový případ vs lokální skryté proměnné",
        "panel1_help": r"""
Tento panel porovnává ideální singletovou korelaci s jednoduchým deterministickým lokálním modelem se skrytými proměnnými.

**Ovladače**
- **Úhlový rozdíl** nastavuje aktuální separaci analyzátorů $\Delta = a-b$ pro snapshot.
- **Počet bodů** řídí rozlišení spojitých křivek.

**Levý graf**
- Ukazuje korelační funkci $E(\Delta)$ v celém intervalu $0 \le \Delta \le \pi$.
- **Kvantová křivka** je kosinusový zákon $-\cos\Delta$.
- **Křivka lokálního modelu** je lomená funkce vznikající v jednoduchém modelu se signovou hranicí. Dává perfektní antikorelaci pro $\Delta=0$, ale neplní celý kvantový kosinus.

**Pravý graf**
- Ukazuje čtyři společné pravděpodobnosti $P(++), P(+-), P(-+), P(--)$ pro zvolený úhlový rozdíl.
- U singletu jsou pravděpodobnosti stejných výsledků potlačeny při zarovnaných analyzátorech, zatímco opačné výsledky dominují.
- Se změnou úhlu se váhy mezi výstupy plynule přelévají.

**Jak to číst**
- Nejde jen o to, že by jedna křivka byla „větší“ než druhá.
- Důležité je, že **funkční tvar** kvantové korelace je dost odlišný na to, aby pro vhodné úhly umožnil porušení CHSH [2–4].
""",
        "angle_difference": "Úhlový rozdíl Δ (stupně)",
        "resolution": "Počet bodů",
        "panel2_title": "2. Průzkumník CHSH",
        "panel2_help": r"""
Tento panel vyhodnocuje CHSH parametr pro čtyři nastavení analyzátorů $(a,a',b,b')$.

**Ovladače**
- Nastav přímo čtyři úhly analyzátorů.
- Tlačítko **Předvolba maximálního kvantového porušení** načte standardní volbu, která dává $|S|=2\sqrt{2}$ pro singlet.

**Levý horní graf**
- Sloupce zobrazují čtyři korelátory vstupující do CHSH kombinace.
- Přerušované čáry vyznačují klasické omezení a aktuální CHSH hodnotu.

**Pravý horní graf**
- 2D tepelná mapa kvantové CHSH hodnoty v závislosti na $a'$ a $b'$, zatímco $a$ a $b$ zůstávají pevné.
- Lze tak snadno vidět, jak zvláštní je optimální oblast.

**Levý dolní graf**
- Stejná CHSH veličina, ale pro srovnávací lokální model se skrytými proměnnými.
- Zůstává uvnitř klasické meze.

**Pravý dolní graf**
- Kompaktní měřidlo aktuální hodnoty $|S|$ s vyznačením klasické meze $2$ a Tsirelsonovy meze $2\sqrt{2}$.

**Jak to číst**
- Bellovo porušení není vlastností libovolné volby úhlů.
- Objevuje se jen pro určité geometrie měřicích nastavení.
- CHSH nerovnost převádí rozdíl mezi lokální a kvantovou strukturou korelací do jedné testovací veličiny [3,4,6].
""",
        "a_angle": "a (stupně)",
        "ap_angle": "a' (stupně)",
        "b_angle": "b (stupně)",
        "bp_angle": "b' (stupně)",
        "set_maximal": "Načíst předvolbu maximálního kvantového porušení",
        "panel3_title": "3. CHSH test po jednotlivých událostech",
        "panel3_help": r"""
Tento panel simuluje Bellův test jako akumulaci mnoha detekovaných párů.

**Ovladače**
- **Počet emitovaných párů** určuje celkový Monte Carlo vzorek.
- **Viditelnost** přimíchává k singletu bílý šum pomocí Wernerovy viditelnosti $V$.
- **Rychlost animace** nastavuje rychlost přehrávání.

**Animovaný graf**
- Plná křivka ukazuje průběžný odhad CHSH hodnoty pro simulovaný kvantový zdroj.
- Tečkovaná křivka ukazuje totéž pro jednoduchý lokální model se skrytými proměnnými.
- Světlé vodorovné čáry označují klasickou mez $2$ a ideální Tsirelsonovu hodnotu $2\sqrt{2}$.

**Boční graf**
- Ukazuje průběžné podmíněné korelace pro čtyři páry nastavení.
- Na začátku silně fluktuují, protože v každém kanálu je málo událostí.
- Později se stabilizují poblíž cílových hodnot.

**Co animace znamená**
- Jednotlivá událost je lokální a diskrétní, ale Bellův parametr se skládá ze **statistiky akumulované přes mnoho událostí**.
- Porušení se neobjeví v jediném páru; vzniká až v souhrnných korelacích [3–5].

**Export GIF**
- Export GIF reprodukuje průběžné budování CHSH hodnoty přes matplotlib a Pillow, takže nezávisí na Chrome ani Kaleido.
""",
        "num_pairs": "Počet emitovaných párů",
        "visibility": "Viditelnost V",
        "animation_speed": "Rychlost animace (ms na snímek)",
        "panel4_title": "4. Šum, viditelnost, provázání a Bellovo porušení",
        "panel4_help": r"""
Tento panel ukazuje, jak přidání bílého šumu mění vztah mezi provázáním a Bellovým porušením.

**Ovladače**
- **Viditelnost** nastavuje směs Wernerova stavu
  $$
  \rho_W = V |\psi^{-}\rangle\langle\psi^{-}| + (1-V)\frac{\mathbb{I}}{4}.
  $$

**Levý graf**
- Zobrazuje zašuměnou korelační funkci $E(\Delta)=-V\cos\Delta$ spolu s klasickou srovnávací křivkou.
- Rostoucí šum zmenšuje amplitudu korelace.

**Pravý graf**
- Ukazuje maximální kvantovou CHSH hodnotu pro Wernerův stav,
  $$
  S_{\max} = 2\sqrt{2}\,V.
  $$
- Vodorovné značky označují klasickou CHSH mez a hranici provázání Wernerova stavu.

**Jak to číst**
- Existuje oblast, kde je Wernerův stav stále provázaný, ale už neporušuje CHSH.
- To pomáhá oddělit dvě myšlenky, které se často zaměňují:
  - **provázání**
  - **Bellova nelokalita**
- Simulátor tento rozdíl ukazuje jediným posuvníkem šumu [4,7,8].
""",
        "reset_panel": "Resetovat panel",
        "download_gif": "Stáhnout GIF",
        "prepare_gif": "Připravit GIF",
        "gif_expander": "Uložit tuto animaci jako GIF",
        "frame_ms": "Délka snímku (ms)",
        "max_frames": "Maximální počet snímků",
        "gif_ready": "GIF je připraven.",
        "quantum": "Kvantový model",
        "lhv": "Lokální model se skrytými proměnnými",
        "classical_bound": "Klasická mez",
        "tsirelson": "Tsirelsonova mez",
        "werner_threshold": "Hranice provázání Wernerova stavu",
    },
}

# ---------------------------
# Utilities
# ---------------------------

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

ket00 = np.array([1, 0, 0, 0], dtype=complex)
ket01 = np.array([0, 1, 0, 0], dtype=complex)
ket10 = np.array([0, 0, 1, 0], dtype=complex)
ket11 = np.array([0, 0, 0, 1], dtype=complex)

psi_minus = (ket01 - ket10) / np.sqrt(2)
rho_singlet = np.outer(psi_minus, psi_minus.conj())
rho_mix = np.eye(4, dtype=complex) / 4.0

def axis_operator(theta_rad):
    return np.cos(theta_rad) * sz + np.sin(theta_rad) * sx

def projector(theta_rad, outcome):
    return 0.5 * (I2 + outcome * axis_operator(theta_rad))

def joint_probabilities(rho, a, b):
    probs = np.zeros((2, 2), dtype=float)
    outcomes = [1, -1]
    for i, sa in enumerate(outcomes):
        Pa = projector(a, sa)
        for j, sb in enumerate(outcomes):
            Pb = projector(b, sb)
            P = np.kron(Pa, Pb)
            probs[i, j] = np.real(np.trace(rho @ P))
    probs = np.clip(probs, 0, None)
    probs /= probs.sum()
    return probs

def correlator_quantum(rho, a, b):
    A = axis_operator(a)
    B = axis_operator(b)
    return float(np.real(np.trace(rho @ np.kron(A, B))))

def rho_werner(V):
    return V * rho_singlet + (1.0 - V) * rho_mix

def fold_delta(delta):
    d = np.mod(np.abs(delta), 2 * np.pi)
    d = np.where(d > np.pi, 2 * np.pi - d, d)
    return d

def correlator_lhv(delta):
    d = fold_delta(delta)
    return -1.0 + 2.0 * d / np.pi

def max_chsh_quantum_for_werner(V):
    return 2.0 * np.sqrt(2.0) * V

def set_default(name, value):
    st.session_state.setdefault(name, value)

def reset_keys(mapping):
    current_lang = st.session_state.get("app_lang", "English")
    for k, v in mapping.items():
        st.session_state[k] = v
    st.session_state["app_lang"] = current_lang
    st.rerun()

@st.cache_data(show_spinner=False)
def generate_chsh_run(a_deg, ap_deg, b_deg, bp_deg, visibility, n_pairs, seed=7):
    rng = np.random.default_rng(seed)
    a, ap, b, bp = np.deg2rad([a_deg, ap_deg, b_deg, bp_deg])
    rho = rho_werner(visibility)
    settings = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=int)
    chosen = rng.integers(0, 4, size=n_pairs)
    setting_labels = settings[chosen]

    # Quantum outcomes sampled from Born probabilities.
    quantum_A = np.empty(n_pairs, dtype=int)
    quantum_B = np.empty(n_pairs, dtype=int)
    angle_pairs = [(a, b), (a, bp), (ap, b), (ap, bp)]
    for idx, (th_a, th_b) in enumerate(angle_pairs):
        mask = chosen == idx
        m = int(mask.sum())
        if m == 0:
            continue
        probs = joint_probabilities(rho, th_a, th_b).reshape(-1)
        cdf = np.cumsum(probs)
        r = rng.random(m)
        samp = np.searchsorted(cdf, r)
        # ordering: ++, +-, -+, --
        quantum_A[mask] = np.take([1, 1, -1, -1], samp)
        quantum_B[mask] = np.take([1, -1, 1, -1], samp)

    # Simple deterministic LHV comparison model.
    lam = rng.uniform(0.0, 2.0 * np.pi, size=n_pairs)
    A0 = np.where(np.cos(lam - a) >= 0, 1, -1)
    A1 = np.where(np.cos(lam - ap) >= 0, 1, -1)
    B0 = -np.where(np.cos(lam - b) >= 0, 1, -1)
    B1 = -np.where(np.cos(lam - bp) >= 0, 1, -1)
    lhv_A = np.where(setting_labels[:, 0] == 0, A0, A1)
    lhv_B = np.where(setting_labels[:, 1] == 0, B0, B1)

    prod_q = quantum_A * quantum_B
    prod_l = lhv_A * lhv_B

    run_data = {}
    for model_name, prod in [("quantum", prod_q), ("lhv", prod_l)]:
        Es = []
        counts_list = []
        for idx in range(4):
            mask = chosen == idx
            counts = np.cumsum(mask.astype(int))
            sums = np.cumsum(mask * prod)
            corr = np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=float), where=counts > 0)
            Es.append(corr)
            counts_list.append(counts)
        E00, E01, E10, E11 = Es
        S = E00 + E01 + E10 - E11
        run_data[model_name] = {
            "E00": E00, "E01": E01, "E10": E10, "E11": E11,
            "S": S, "counts": counts_list
        }
    return run_data

def make_running_chsh_figure(run_data, frame_idx, labels, visibility):
    x = np.arange(1, frame_idx + 2)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Running CHSH", "Running correlators"))
    # Left: running S
    S_q = run_data["quantum"]["S"][: frame_idx + 1]
    S_l = run_data["lhv"]["S"][: frame_idx + 1]
    fig.add_trace(go.Scatter(x=x, y=S_q, mode="lines", name=labels["quantum"], line=dict(width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=S_l, mode="lines", name=labels["lhv"], line=dict(width=2, dash="dot")), row=1, col=1)
    fig.add_hline(y=2.0, line_dash="dash", line_color="gray", row=1, col=1, annotation_text=labels["classical_bound"])
    fig.add_hline(y=2*np.sqrt(2)*visibility, line_dash="dashdot", line_color="gray", row=1, col=1, annotation_text=labels["tsirelson"])
    fig.update_xaxes(title_text="Pairs", row=1, col=1)
    fig.update_yaxes(title_text="S", row=1, col=1, range=[-3.0, 3.0])

    names = ["E(a,b)", "E(a,b')", "E(a',b)", "E(a',b')"]
    for name, key in zip(names, ["E00", "E01", "E10", "E11"]):
        fig.add_trace(go.Scatter(x=x, y=run_data["quantum"][key][: frame_idx + 1], mode="lines", name=f"{labels['quantum']} {name}"), row=1, col=2)
    fig.update_xaxes(title_text="Pairs", row=1, col=2)
    fig.update_yaxes(title_text="E", row=1, col=2, range=[-1.05, 1.05])
    fig.update_layout(height=480, margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", y=-0.15))
    return fig

def build_chsh_frames(run_data, labels, visibility, n_frames=80):
    N = len(run_data["quantum"]["S"])
    idx = np.linspace(0, N - 1, n_frames, dtype=int)
    frames = []
    for k in idx:
        subfig = make_running_chsh_figure(run_data, k, labels, visibility)
        frames.append(go.Frame(data=subfig.data, name=str(k)))
    return idx, frames

def render_plotly_animation(run_data, labels, visibility, speed_ms):
    N = len(run_data["quantum"]["S"])
    frame_idx, frames = build_chsh_frames(run_data, labels, visibility)
    fig = make_running_chsh_figure(run_data, frame_idx[0], labels, visibility)
    fig.frames = frames
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "Play", "method": "animate",
                 "args": [None, {"frame": {"duration": int(speed_ms), "redraw": True},
                                 "transition": {"duration": 0},
                                 "fromcurrent": True}]},
                {"label": "Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate",
                                   "transition": {"duration": 0}}]},
            ],
            "direction": "left",
            "x": 0.01,
            "y": 1.15,
            "showactive": False,
        }]
    )
    st.plotly_chart(fig, use_container_width=True)

def gif_running_chsh(run_data, labels, visibility, frame_ms=120, max_frames=80):
    N = len(run_data["quantum"]["S"])
    idx = np.linspace(0, N - 1, min(max_frames, N), dtype=int)
    images = []
    for k in idx:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=140)
        x = np.arange(1, k + 2)

        axes[0].plot(x, run_data["quantum"]["S"][:k+1], lw=2.5, label=labels["quantum"])
        axes[0].plot(x, run_data["lhv"]["S"][:k+1], lw=1.8, ls="--", label=labels["lhv"])
        axes[0].axhline(2.0, color="gray", ls=":")
        axes[0].axhline(2*np.sqrt(2)*visibility, color="gray", ls="-.")
        axes[0].set_title("Running CHSH")
        axes[0].set_xlabel("Pairs")
        axes[0].set_ylabel("S")
        axes[0].set_ylim(-3.0, 3.0)
        axes[0].legend(loc="lower right", fontsize=8)

        for name, key in zip(["E(a,b)", "E(a,b')", "E(a',b)", "E(a',b')"], ["E00", "E01", "E10", "E11"]):
            axes[1].plot(x, run_data["quantum"][key][:k+1], lw=1.7, label=name)
        axes[1].set_title("Running correlators")
        axes[1].set_xlabel("Pairs")
        axes[1].set_ylabel("E")
        axes[1].set_ylim(-1.05, 1.05)
        axes[1].legend(loc="lower right", fontsize=8)

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))

    out = io.BytesIO()
    images[0].save(out, format="GIF", save_all=True, append_images=images[1:], duration=int(frame_ms), loop=0)
    out.seek(0)
    return out

# ---------------------------
# Session state defaults
# ---------------------------

set_default("app_lang", "English")
set_default("p1_delta_deg", 45.0)
set_default("p1_npts", 361)
set_default("p2_a", 0.0)
set_default("p2_ap", 90.0)
set_default("p2_b", 45.0)
set_default("p2_bp", -45.0)
set_default("p3_pairs", 1200)
set_default("p3_visibility", 1.0)
set_default("p3_speed", 120)
set_default("p4_visibility", 0.85)

# ---------------------------
# Sidebar
# ---------------------------

st.title(TEXT["en"]["title"] if st.session_state["app_lang"] == "English" else TEXT["cs"]["title"])
lang_label = st.sidebar.selectbox(TEXT["en"]["language"], ["English", "Čeština"], index=0 if st.session_state["app_lang"] == "English" else 1, key="app_lang")
lang = "en" if lang_label == "English" else "cs"
T = TEXT[lang]
st.caption(T["subtitle"])
st.info(T["global_help"])

with st.expander(T["theory_title"], expanded=False):
    st.markdown(T["theory_body"])

# ---------------------------
# Panel 1
# ---------------------------

st.header(T["panel1_title"])
with st.expander("How to use this panel / Jak používat tento panel", expanded=False):
    st.markdown(T["panel1_help"])

col_ctrl, col_plot = st.columns([1, 3], gap="large")
with col_ctrl:
    st.slider(T["angle_difference"], 0.0, 180.0, key="p1_delta_deg")
    st.slider(T["resolution"], 121, 721, step=20, key="p1_npts")
    if st.button(T["reset_panel"], key="reset_p1"):
        reset_keys({"p1_delta_deg": 45.0, "p1_npts": 361})

with col_plot:
    npts = int(st.session_state["p1_npts"])
    deltas = np.linspace(0.0, np.pi, npts)
    qcorr = -np.cos(deltas)
    lcorr = correlator_lhv(deltas)

    dsel = np.deg2rad(st.session_state["p1_delta_deg"])
    probs_sel = joint_probabilities(rho_singlet, 0.0, dsel)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.rad2deg(deltas), y=qcorr, mode="lines", name=T["quantum"], line=dict(width=3)))
        fig.add_trace(go.Scatter(x=np.rad2deg(deltas), y=lcorr, mode="lines", name=T["lhv"], line=dict(width=2, dash="dot")))
        fig.add_vline(x=float(st.session_state["p1_delta_deg"]), line_dash="dash", line_color="gray")
        fig.update_layout(height=420, xaxis_title="Δ (degrees)", yaxis_title="E(Δ)", margin=dict(l=20, r=20, t=20, b=20))
        fig.update_yaxes(range=[-1.05, 1.05])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        labels = ["++", "+−", "−+", "−−"]
        vals = [probs_sel[0,0], probs_sel[0,1], probs_sel[1,0], probs_sel[1,1]]
        fig2 = go.Figure(go.Bar(x=labels, y=vals))
        fig2.update_layout(height=420, xaxis_title="Joint outcomes", yaxis_title="Probability", margin=dict(l=20, r=20, t=20, b=20))
        fig2.update_yaxes(range=[0, 1])
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# Panel 2
# ---------------------------

st.header(T["panel2_title"])
with st.expander("How to use this panel / Jak používat tento panel", expanded=False):
    st.markdown(T["panel2_help"])

col_ctrl, col_plot = st.columns([1.1, 3], gap="large")
with col_ctrl:
    st.slider(T["a_angle"], -180.0, 180.0, key="p2_a")
    st.slider(T["ap_angle"], -180.0, 180.0, key="p2_ap")
    st.slider(T["b_angle"], -180.0, 180.0, key="p2_b")
    st.slider(T["bp_angle"], -180.0, 180.0, key="p2_bp")
    if st.button(T["set_maximal"], key="set_max_p2"):
        st.session_state["p2_a"] = 0.0
        st.session_state["p2_ap"] = 90.0
        st.session_state["p2_b"] = 45.0
        st.session_state["p2_bp"] = -45.0
        st.rerun()
    if st.button(T["reset_panel"], key="reset_p2"):
        reset_keys({"p2_a": 0.0, "p2_ap": 90.0, "p2_b": 45.0, "p2_bp": -45.0})

with col_plot:
    a, ap, b, bp = np.deg2rad([st.session_state["p2_a"], st.session_state["p2_ap"], st.session_state["p2_b"], st.session_state["p2_bp"]])
    rho = rho_singlet
    E00_q = correlator_quantum(rho, a, b)
    E01_q = correlator_quantum(rho, a, bp)
    E10_q = correlator_quantum(rho, ap, b)
    E11_q = correlator_quantum(rho, ap, bp)
    S_q = E00_q + E01_q + E10_q - E11_q

    E00_l = correlator_lhv(a - b)
    E01_l = correlator_lhv(a - bp)
    E10_l = correlator_lhv(ap - b)
    E11_l = correlator_lhv(ap - bp)
    S_l = E00_l + E01_l + E10_l - E11_l

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        labels_corr = ["E(a,b)", "E(a,b')", "E(a',b)", "E(a',b')"]
        fig.add_trace(go.Bar(x=labels_corr, y=[E00_q, E01_q, E10_q, E11_q], name=T["quantum"]))
        fig.add_trace(go.Bar(x=labels_corr, y=[E00_l, E01_l, E10_l, E11_l], name=T["lhv"]))
        fig.update_layout(barmode="group", height=360, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Correlation")
        fig.update_yaxes(range=[-1.05, 1.05])
        st.plotly_chart(fig, use_container_width=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=abs(S_q),
            number={"suffix": ""},
            gauge={
                "axis": {"range": [0, 3.0]},
                "steps": [{"range": [0, 2], "color": "#D3D3D3"},
                          {"range": [2, 2*np.sqrt(2)], "color": "#B0E0E6"}],
                "threshold": {"line": {"color": "black", "width": 3}, "value": 2*np.sqrt(2)}
            },
            title={"text": "|S|"}
        ))
        fig_gauge.add_annotation(x=0.5, y=0.05, text=f"{T['classical_bound']}: 2<br>{T['tsirelson']}: 2√2", showarrow=False)
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with c2:
        grid = np.linspace(-180, 180, 181)
        AP, BP = np.meshgrid(np.deg2rad(grid), np.deg2rad(grid), indexing="xy")
        Smap_q = -np.cos(a - b) - np.cos(a - BP) - np.cos(AP - b) + np.cos(AP - BP)
        Smap_l = correlator_lhv(a - b) + correlator_lhv(a - BP) + correlator_lhv(AP - b) - correlator_lhv(AP - BP)

        fig_h1 = go.Figure(go.Heatmap(x=grid, y=grid, z=Smap_q, colorbar_title="S"))
        fig_h1.add_scatter(x=[np.rad2deg(bp)], y=[np.rad2deg(ap)], mode="markers", marker=dict(size=10, color="white", symbol="x"))
        fig_h1.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="b' (deg)", yaxis_title="a' (deg)")
        st.plotly_chart(fig_h1, use_container_width=True)

        fig_h2 = go.Figure(go.Heatmap(x=grid, y=grid, z=Smap_l, colorbar_title="S"))
        fig_h2.add_scatter(x=[np.rad2deg(bp)], y=[np.rad2deg(ap)], mode="markers", marker=dict(size=10, color="white", symbol="x"))
        fig_h2.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="b' (deg)", yaxis_title="a' (deg)")
        st.plotly_chart(fig_h2, use_container_width=True)

# ---------------------------
# Panel 3
# ---------------------------

st.header(T["panel3_title"])
with st.expander("How to use this panel / Jak používat tento panel", expanded=False):
    st.markdown(T["panel3_help"])

col_ctrl, col_plot = st.columns([1.1, 3], gap="large")
with col_ctrl:
    st.slider(T["num_pairs"], 200, 5000, step=100, key="p3_pairs")
    st.slider(T["visibility"], 0.0, 1.0, step=0.01, key="p3_visibility")
    st.slider(T["animation_speed"], 40, 400, step=10, key="p3_speed")
    if st.button(T["reset_panel"], key="reset_p3"):
        reset_keys({"p3_pairs": 1200, "p3_visibility": 1.0, "p3_speed": 120})

run_data = generate_chsh_run(
    st.session_state["p2_a"],
    st.session_state["p2_ap"],
    st.session_state["p2_b"],
    st.session_state["p2_bp"],
    st.session_state["p3_visibility"],
    int(st.session_state["p3_pairs"]),
)

with col_plot:
    render_plotly_animation(run_data, {
        "quantum": T["quantum"],
        "lhv": T["lhv"],
        "classical_bound": T["classical_bound"],
        "tsirelson": T["tsirelson"],
    }, st.session_state["p3_visibility"], st.session_state["p3_speed"])

    with st.expander(T["gif_expander"], expanded=False):
        frame_ms = st.slider(T["frame_ms"], 40, 400, 120, 10, key="p3_frame_ms")
        max_frames = st.slider(T["max_frames"], 20, 180, 80, 5, key="p3_max_frames")
        if st.button(T["prepare_gif"], key="prep_p3_gif"):
            gif_buf = gif_running_chsh(run_data, {
                "quantum": T["quantum"],
                "lhv": T["lhv"],
                "classical_bound": T["classical_bound"],
                "tsirelson": T["tsirelson"],
            }, st.session_state["p3_visibility"], frame_ms=frame_ms, max_frames=max_frames)
            st.session_state["p3_gif"] = gif_buf.getvalue()
            st.success(T["gif_ready"])
        if "p3_gif" in st.session_state:
            st.download_button(T["download_gif"], data=st.session_state["p3_gif"], file_name="bell_chsh_running.gif", mime="image/gif", key="download_p3_gif")

# ---------------------------
# Panel 4
# ---------------------------

st.header(T["panel4_title"])
with st.expander("How to use this panel / Jak používat tento panel", expanded=False):
    st.markdown(T["panel4_help"])

col_ctrl, col_plot = st.columns([1.1, 3], gap="large")
with col_ctrl:
    st.slider(T["visibility"], 0.0, 1.0, step=0.01, key="p4_visibility")
    if st.button(T["reset_panel"], key="reset_p4"):
        reset_keys({"p4_visibility": 0.85})

with col_plot:
    V = st.session_state["p4_visibility"]
    deltas = np.linspace(0.0, np.pi, 361)
    qcorr = -V * np.cos(deltas)
    lcorr = correlator_lhv(deltas)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.rad2deg(deltas), y=qcorr, mode="lines", name=f"{T['quantum']} (V={V:.2f})", line=dict(width=3)))
        fig.add_trace(go.Scatter(x=np.rad2deg(deltas), y=lcorr, mode="lines", name=T["lhv"], line=dict(width=2, dash="dot")))
        fig.update_layout(height=420, xaxis_title="Δ (degrees)", yaxis_title="E(Δ)", margin=dict(l=20, r=20, t=20, b=20))
        fig.update_yaxes(range=[-1.05, 1.05])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        Vs = np.linspace(0.0, 1.0, 401)
        Smax = max_chsh_quantum_for_werner(Vs)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=Vs, y=Smax, mode="lines", name="Smax"))
        fig2.add_hline(y=2.0, line_dash="dash", annotation_text=T["classical_bound"])
        fig2.add_vline(x=1/3, line_dash="dot", annotation_text=T["werner_threshold"])
        fig2.add_scatter(x=[V], y=[max_chsh_quantum_for_werner(V)], mode="markers", marker=dict(size=11), name=f"V={V:.2f}")
        fig2.update_layout(height=420, xaxis_title="V", yaxis_title="Maximum CHSH value", margin=dict(l=20, r=20, t=20, b=20))
        fig2.update_yaxes(range=[0, 3.1])
        st.plotly_chart(fig2, use_container_width=True)
