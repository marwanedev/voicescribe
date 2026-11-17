r"""
Phase 2 — Bilingual (EN/FR) Dataset Builder (`scripts/build_dataset.py`)
Run with:
    .\.venv\Scripts\python.exe scripts/build_dataset.py

Generates a reproducible, balanced 200-clip speech dataset (100 English + 100 French):
- Uses installed Windows SAPI speech synthesizers (`en-US` David & Zira, `fr-FR` Hortense)
  across varied speaking rates (-2 to +2) and controlled acoustic variations (pitch/formant
  shifts & subtle room ambience) so clips exhibit natural variance in duration, cadence,
  spectral envelope, and MFCC distributions.
- Saves WAV files under `data/raw/en/` and `data/raw/fr/`.
- Writes `data/metadata.csv` with an 80/20 stratified split:
  - `train`: 160 clips (80 EN + 80 FR) for training the custom MFCC classifier.
  - `test` :  40 clips (20 EN + 20 FR) for head-to-head evaluation (MFCC vs. Whisper).
"""

import csv
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import librosa
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_utils import SAMPLE_RATE, get_audio_stats, load_and_preprocess_audio

RAW_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_CSV = PROJECT_ROOT / "data" / "metadata.csv"

ENGLISH_SENTENCES: List[str] = [
    "Artificial intelligence is transforming modern healthcare and medical diagnostics.",
    "Automatic speech recognition converts spoken language into readable text.",
    "Please remember to upload your project report before the Friday deadline.",
    "Deep learning models rely on large datasets and neural network architectures.",
    "The train to London departs from platform four in fifteen minutes.",
    "Accurate subtitles make educational videos accessible to students worldwide.",
    "We are testing how different model sizes affect transcription speed.",
    "Machine learning algorithms can identify patterns in complex acoustic signals.",
    "Good morning everyone and welcome to our university presentation.",
    "Signal processing starts by sampling continuous sound waves at sixteen kilohertz.",
    "Can you tell me where the nearest library is located on campus?",
    "The weather forecast predicts heavy rain and strong winds tomorrow afternoon.",
    "Neural networks use attention mechanisms to focus on relevant audio frames.",
    "Our research question compares word error rate across multiple architectures.",
    "I would like to order a cup of hot coffee and a croissant.",
    "Voice activity detection removes silent pauses before decoding speech.",
    "Renewable energy sources like solar and wind power are growing rapidly.",
    "This experiment evaluates classification precision recall and area under the curve.",
    "Listening carefully to customer feedback helps engineers build better software.",
    "The museum opens at nine in the morning every Tuesday through Sunday.",
    "Spectrograms represent how signal frequencies change over short time windows.",
    "Students must understand every line of code they present in class.",
    "Reading academic papers provides strong context for experimental findings.",
    "High background noise can increase transcription errors in small models.",
    "Let us schedule a team meeting tomorrow to review the latest benchmarks.",
    "Digital audio workstations allow editors to synchronize captions with video.",
    "Computer vision and natural language processing are major branches of AI.",
    "The conference featured keynote speakers from leading research laboratories.",
    "Data preprocessing is often the most critical step in any pipeline.",
    "We split our dataset into eighty percent training and twenty percent testing.",
    "Random forest classifiers combine predictions from many decision trees.",
    "Whisper uses an encoder decoder transformer trained on multilingual audio.",
    "Please speak clearly into the microphone so the system captures your voice.",
    "The university library offers quiet study rooms for group projects.",
    "Cloud computing enables scalable storage but local inference protects privacy.",
    "Every subtitle block contains a sequence number start time and end time.",
    "Scientists discovered a new exoplanet orbiting a distant red dwarf star.",
    "Regular exercise and balanced nutrition improve long term physical health.",
    "Version control with Git helps teammates collaborate without losing code.",
    "The symphony orchestra performed a prachtige classical concert last night.",
    "Greedy decoding selects the highest probability token at each time step.",
    "Beam search explores multiple candidate sequences to improve accuracy.",
    "Feature extraction compresses raw waveforms into compact numerical vectors.",
    "Cross validation prevents overfitting when tuning model hyperparameters.",
    "The flight was delayed by two hours due to dense morning fog.",
    "Modern smartphones include dedicated processors for neural network tasks.",
    "Cybersecurity experts recommend using strong passwords and two factor authentication.",
    "We observed a clear trade off between inference latency and accuracy.",
    "Open source software accelerates scientific discovery and reproducibility.",
    "The botanical garden contains hundreds of rare tropical plant species.",
    "Acoustic modeling maps audio frames to phonetic or subword units.",
    "Language identification predicts which language a person is speaking.",
    "The restaurant on the corner serves delicious Mediterranean dishes.",
    "Standardizing features to zero mean and unit variance helps neural networks converge.",
    "Our web application is built with Python and the Streamlit framework.",
    "Real time factor measures how fast a model processes one second of audio.",
    "The robotics club won first prize at the national engineering competition.",
    "Clean documentation makes it easy for new developers to join a project.",
    "Ocean currents play a vital role in regulating global climate systems.",
    "Logistic regression estimates class probabilities using the sigmoid function.",
    "The bookstore just received a shipment of new computer science textbooks.",
    "Short time Fourier transform decomposes a signal into frequency components.",
    "Multilingual systems allow people from different countries to communicate.",
    "We verified that all python packages match the exact same version.",
    "The documentary explored the history of aviation over the past century.",
    "Confusion matrices show true positives false positives and false negatives.",
    "Audio normalization ensures consistent volume levels across recordings.",
    "The express bus goes directly from the airport to the city center.",
    "Hyperparameter tuning improved our validation score by four percent.",
    "Thank you for listening to our presentation and we welcome your questions.",
    "Electric vehicles are becoming more affordable as battery technology improves.",
    "The human ear perceives pitch logarithmically rather than linearly.",
    "Mel frequency cepstral coefficients capture vocal tract resonance characteristics.",
    "We exported the final transcript as both plain text and subtitle files.",
    "The chemistry laboratory requires protective glasses and lab coats at all times.",
    "Automated grading systems must be tested carefully for fairness and accuracy.",
    "Satellite imagery helps farmers monitor crop health and soil moisture.",
    "The software update fixed several memory leaks and improved stability.",
    "Learning a second language strengthens cognitive flexibility and memory.",
    "The marathon runners crossed the finish line despite the warm weather.",
    "Matrix multiplication is the core mathematical operation in deep learning.",
    "Our baseline model runs in less than one second on a standard laptop.",
    "The historical archive preserves ancient manuscripts and rare photographs.",
    "Quantization reduces model memory footprint by using eight bit integers.",
    "Please check your microphone connection before starting the live recording.",
    "The astronomy club organized a stargazing event on top of the hill.",
    "Descriptive statistics include the mean median mode and standard deviation.",
    "Zero shot learning allows a model to handle tasks without extra training.",
    "The bridge was designed to withstand strong earthquakes and heavy traffic.",
    "Subword tokenization helps language models handle rare and unknown words.",
    "We plotted the receiver operating characteristic curve to evaluate performance.",
    "The art exhibition showcased paintings from the nineteenth century.",
    "Overfitting happens when a model memorizes training noise instead of patterns.",
    "Audio compression formats like MP3 reduce file size by removing imperceptible sounds.",
    "The mountain trail offers breathtaking views of the surrounding valley.",
    "Self attention layers compute relationships between all positions in a sequence.",
    "Our experimental results confirm the findings reported in recent literature.",
    "The university campus is vibrant during the autumn semester.",
    "Speech synthesis and speech recognition are complementary technologies.",
    "We are proud to present the final version of the VoiceScribe system.",
]

FRENCH_SENTENCES: List[str] = [
    "L'intelligence artificielle transforme la médecine moderne et le diagnostic clinique.",
    "La reconnaissance vocale automatique convertit la parole en texte lisible.",
    "N'oubliez pas d'envoyer votre rapport de projet avant la date limite de vendredi.",
    "Les modèles d'apprentissage profond reposent sur de grands ensembles de données.",
    "Le train à destination de Paris partira du quai numéro quatre dans quinze minutes.",
    "Des sous-titres précis rendent les vidéos éducatives accessibles à tous les étudiants.",
    "Nous testons l'impact de la taille du modèle sur la vitesse de transcription.",
    "Les algorithmes d'apprentissage automatique détectent des motifs dans les signaux audio.",
    "Bonjour à tous et bienvenue à notre présentation universitaire.",
    "Le traitement du signal commence par un échantillonnage à seize kilohertz.",
    "Pouvez-vous m'indiquer où se trouve la bibliothèque la plus proche sur le campus ?",
    "Les prévisions météorologiques annoncent de fortes pluies demain après-midi.",
    "Les réseaux de neurones utilisent le mécanisme d'attention pour analyser l'audio.",
    "Notre question de recherche compare le taux d'erreur sur les mots entre plusieurs modèles.",
    "Je voudrais commander une tasse de café chaud et un croissant s'il vous plaît.",
    "La détection d'activité vocale supprime les silences avant la transcription.",
    "Les énergies renouvelables comme le solaire et l'éolien se développent rapidement.",
    "Cette expérience évalue la précision le rappel et l'aire sous la courbe.",
    "Écouter attentivement les utilisateurs aide les ingénieurs à créer de meilleurs logiciels.",
    "Le musée ouvre ses portes à neuf heures du matin du mardi au dimanche.",
    "Les spectrogrammes montrent comment les fréquences évoluent au cours du temps.",
    "Les étudiants doivent comprendre chaque ligne de code présentée en classe.",
    "La lecture d'articles scientifiques offre une base solide pour nos expériences.",
    "Un bruit de fond élevé peut augmenter les erreurs de transcription.",
    "Organisons une réunion d'équipe demain pour analyser les derniers résultats.",
    "Les logiciels de montage permettent de synchroniser les sous-titres avec la vidéo.",
    "La vision par ordinateur et le traitement du langage naturel sont essentiels.",
    "La conférence a réuni des chercheurs issus des meilleurs laboratoires internationaux.",
    "Le prétraitement des données est souvent l'étape la plus importante du projet.",
    "Nous avons divisé notre jeu de données en quatre-vingts pour cent d'entraînement.",
    "Les forêts aléatoires combinent les prédictions de nombreux arbres de décision.",
    "Le modèle Whisper utilise une architecture transformeur entraînée sur de nombreuses langues.",
    "Veuillez parler clairement dans le microphone pour enregistrer votre voix.",
    "La bibliothèque universitaire propose des salles calmes pour le travail en groupe.",
    "Le calcul local sur ordinateur portable garantit la confidentialité des données.",
    "Chaque bloc de sous-titre contient un numéro une heure de début et de fin.",
    "Les astronomes ont découvert une nouvelle exoplanète autour d'une étoile lointaine.",
    "L'exercice physique régulier et une alimentation équilibrée améliorent la santé.",
    "La gestion de versions avec Git facilite le travail collaboratif entre étudiants.",
    "L'orchestre symphonique a donné un magnifique concert classique hier soir.",
    "Le décodage glouton choisit le mot le plus probable à chaque étape.",
    "La recherche en faisceau explore plusieurs phrases candidates simultanément.",
    "L'extraction de caractéristiques transforme le signal brut en vecteurs numériques.",
    "La validation croisée évite le surapprentissage lors du réglage des paramètres.",
    "Le vol a été retardé de deux heures à cause d'un épais brouillard matinal.",
    "Les téléphones modernes intègrent des processeurs dédiés aux réseaux de neurones.",
    "Les experts en sécurité recommandent l'utilisation de mots de passe robustes.",
    "Nous observons un compromis clair entre le temps de calcul et la précision.",
    "Les logiciels libres accélèrent la recherche scientifique et la reproductibilité.",
    "Le jardin botanique abrite des centaines d'espèces de plantes tropicales rares.",
    "La modélisation acoustique relie les trames audio aux unités phonétiques.",
    "L'identification de la langue prédit automatiquement la langue parlée.",
    "Le petit restaurant du quartier prépare d'excellents plats traditionnels.",
    "La normalisation des données aide les algorithmes d'apprentissage à converger.",
    "Notre interface utilisateur interactive est développée avec Python et Streamlit.",
    "Le facteur temps réel mesure la vitesse de traitement par seconde d'audio.",
    "Le club de robotique a remporté le premier prix du concours national.",
    "Une documentation claire facilite la compréhension de chaque module du code.",
    "Les courants océaniques jouent un rôle majeur dans la régulation du climat.",
    "La régression logistique estime les probabilités à l'aide d'une fonction sigmoïde.",
    "La librairie vient de recevoir de nouveaux manuels d'informatique et de mathématiques.",
    "La transformée de Fourier décompose un signal sonore en différentes fréquences.",
    "Les systèmes multilingues facilitent la communication entre différentes cultures.",
    "Nous avons vérifié que toutes les bibliothèques utilisent exactement la même version.",
    "Le documentaire retrace l'histoire fascinante de l'aviation au vingtième siècle.",
    "La matrice de confusion résume les prédictions correctes et les erreurs du modèle.",
    "La normalisation audio assure un volume sonore homogène sur tous les fichiers.",
    "La navette rapide relie directement l'aéroport au centre de la ville.",
    "L'optimisation des hyperparamètres a amélioré nos performances de quatre pour cent.",
    "Merci beaucoup pour votre attention et nous sommes prêts à répondre à vos questions.",
    "Les véhicules électriques deviennent plus accessibles grâce aux nouvelles batteries.",
    "L'oreille humaine perçoit les fréquences sonores sur une échelle logarithmique.",
    "Les coefficients cepstraux capturent les caractéristiques acoustiques de la voix.",
    "Nous avons exporté la transcription finale au format texte et au format sous-titre.",
    "Le laboratoire de chimie impose le port de lunettes de protection.",
    "Les systèmes automatisés doivent être évalués avec rigueur et objectivité.",
    "Les images satellites aident les agriculteurs à surveiller l'état des cultures.",
    "La dernière mise à jour logicielle a corrigé plusieurs erreurs et amélioré la stabilité.",
    "Apprendre une langue étrangère développe la mémoire et la souplesse d'esprit.",
    "Les coureurs du marathon ont franchi la ligne d'arrivée sous les applaudissements.",
    "La multiplication matricielle est l'opération fondamentale des réseaux profonds.",
    "Notre modèle de classification s'exécute en moins d'une seconde sur le processeur.",
    "Les archives nationales conservent des manuscrits anciens et des photographies historiques.",
    "La quantification réduit la mémoire nécessaire en utilisant des entiers sur huit bits.",
    "Vérifiez que votre microphone est bien branché avant de lancer l'enregistrement.",
    "Le club d'astronomie a organisé une soirée d'observation des étoiles sur la colline.",
    "Les statistiques descriptives incluent la moyenne la médiane le mode et l'écart type.",
    "L'apprentissage sans exemple permet au modèle de généraliser à de nouvelles tâches.",
    "Ce pont suspendu a été conçu pour résister aux vents violents et au trafic dense.",
    "La segmentation en sous-mots permet de traiter efficacement les mots rares.",
    "Nous avons tracé la courbe caractéristique de performance pour comparer les modèles.",
    "L'exposition artistique présentait des tableaux célèbres du dix-neuvième siècle.",
    "Le surapprentissage survient lorsque le modèle mémorise le bruit d'entraînement.",
    "Les formats compressés comme le MP3 réduisent la taille des fichiers audio.",
    "Le sentier de montagne offre une vue panoramique exceptionnelle sur la vallée.",
    "Les couches d'attention calculent les relations entre tous les mots de la phrase.",
    "Nos résultats expérimentaux confirment les conclusions des travaux récents.",
    "Le campus universitaire est très animé pendant le semestre d'automne.",
    "La synthèse vocale et la reconnaissance vocale sont deux technologies complémentaires.",
    "Nous sommes ravis de vous présenter la version complète du système VoiceScribe.",
]


def synthesize_batch_powershell(items: list[dict]) -> None:
    """Invoke Windows SAPI once in PowerShell to synthesize all missing WAV files quickly."""
    missing = [it for it in items if not Path(it["abs_path"]).exists()]
    if not missing:
        return

    print(f"Synthesizing {len(missing)} audio clips via Windows Speech Synthesizer...")
    lines = [
        "Add-Type -AssemblyName System.Speech",
        "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer",
    ]
    for it in missing:
        voice_name = it["voice"]
        rate = it["rate"]
        out_path = str(it["abs_path"]).replace("'", "''")
        text = str(it["transcript"]).replace("'", "''")
        lines.append(f"$synth.SelectVoice('{voice_name}')")
        lines.append(f"$synth.Rate = {rate}")
        lines.append(f"$synth.SetOutputToWaveFile('{out_path}')")
        lines.append(f"$synth.Speak('{text}')")

    lines.append("$synth.Dispose()")
    ps_command = "\n".join(lines)
    ps_file = Path(tempfile.gettempdir()) / "voicescribe_synth_batch.ps1"
    ps_file.write_text(ps_command, encoding="utf-8-sig")
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps_file)],
        check=True,
    )


def apply_acoustic_variation(
    wav_path: Path, pitch_semitones: float, noise_std: float, seed: int
) -> None:
    """
    Apply subtle, realistic speaker pitch variation (`pitch_shift`) and mild room floor
    noise so the 200 clips represent diverse speaker vocal tracts and recording conditions.
    """
    waveform = load_and_preprocess_audio(wav_path, target_sr=SAMPLE_RATE)
    if abs(pitch_semitones) > 0.05:
        waveform = librosa.effects.pitch_shift(
            waveform, sr=SAMPLE_RATE, n_steps=pitch_semitones
        )
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        waveform = waveform + rng.normal(0.0, noise_std, size=waveform.shape).astype(
            np.float32
        )
    # Normalize peak safely to [-0.95, 0.95]
    peak = float(np.max(np.abs(waveform))) if len(waveform) > 0 else 1.0
    if peak > 0.95:
        waveform = waveform * (0.95 / peak)
    sf.write(str(wav_path), waveform.astype(np.float32), SAMPLE_RATE)


def main() -> None:
    en_dir = RAW_DIR / "en"
    fr_dir = RAW_DIR / "fr"
    en_dir.mkdir(parents=True, exist_ok=True)
    fr_dir.mkdir(parents=True, exist_ok=True)

    en_voices = ["Microsoft Zira Desktop", "Microsoft David Desktop"]
    fr_voices = ["Microsoft Hortense Desktop"]
    rates = [-2, -1, 0, 1, 2]
    pitch_shifts = [-1.5, -0.8, 0.0, 0.8, 1.5]
    noise_levels = [0.0005, 0.001, 0.002, 0.003]

    items = []

    # 100 English clips (first 80 -> train, last 20 -> test)
    for i, text in enumerate(ENGLISH_SENTENCES):
        clip_id = f"en_{i + 1:03d}"
        split = "train" if i < 80 else "test"
        rel_path = Path("data") / "raw" / "en" / f"{clip_id}.wav"
        items.append(
            {
                "clip_id": clip_id,
                "rel_path": rel_path.as_posix(),
                "abs_path": PROJECT_ROOT / rel_path,
                "language": "en",
                "language_name": "English",
                "label": 0,
                "voice": en_voices[i % len(en_voices)],
                "rate": rates[i % len(rates)],
                "pitch": pitch_shifts[(i // 2) % len(pitch_shifts)],
                "noise": noise_levels[i % len(noise_levels)],
                "split": split,
                "transcript": text,
                "seed": 1000 + i,
            }
        )

    # 100 French clips (first 80 -> train, last 20 -> test)
    for i, text in enumerate(FRENCH_SENTENCES):
        clip_id = f"fr_{i + 1:03d}"
        split = "train" if i < 80 else "test"
        rel_path = Path("data") / "raw" / "fr" / f"{clip_id}.wav"
        items.append(
            {
                "clip_id": clip_id,
                "rel_path": rel_path.as_posix(),
                "abs_path": PROJECT_ROOT / rel_path,
                "language": "fr",
                "language_name": "French",
                "label": 1,
                "voice": fr_voices[0],
                "rate": rates[i % len(rates)],
                "pitch": pitch_shifts[(i // 3) % len(pitch_shifts)],
                "noise": noise_levels[i % len(noise_levels)],
                "split": split,
                "transcript": text,
                "seed": 2000 + i,
            }
        )

    need_variation = [it for it in items if not Path(it["abs_path"]).exists()]
    synthesize_batch_powershell(items)

    if need_variation:
        print("Applying speaker pitch & room acoustic diversity and standardizing to 16 kHz WAV...")
        for it in need_variation:
            apply_acoustic_variation(
                Path(it["abs_path"]),
                pitch_semitones=it["pitch"],
                noise_std=it["noise"],
                seed=it["seed"],
            )

    print("Computing clip metadata and writing `data/metadata.csv`...")
    METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "file_path",
        "language",
        "language_name",
        "label",
        "split",
        "voice",
        "rate_setting",
        "pitch_shift_st",
        "duration_sec",
        "word_count",
        "char_count",
        "speaking_rate_wps",
        "rms_energy",
        "peak_amplitude",
        "transcript",
    ]

    with METADATA_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for it in items:
            waveform = load_and_preprocess_audio(it["abs_path"], target_sr=SAMPLE_RATE)
            stats = get_audio_stats(waveform, sr=SAMPLE_RATE)
            words = it["transcript"].split()
            word_count = len(words)
            char_count = len(it["transcript"])
            dur = max(stats["duration_sec"], 0.1)
            wps = round(word_count / dur, 3)

            writer.writerow(
                {
                    "clip_id": it["clip_id"],
                    "file_path": it["rel_path"],
                    "language": it["language"],
                    "language_name": it["language_name"],
                    "label": it["label"],
                    "split": it["split"],
                    "voice": it["voice"],
                    "rate_setting": it["rate"],
                    "pitch_shift_st": it["pitch"],
                    "duration_sec": stats["duration_sec"],
                    "word_count": word_count,
                    "char_count": char_count,
                    "speaking_rate_wps": wps,
                    "rms_energy": stats["rms_energy"],
                    "peak_amplitude": stats["peak_amplitude"],
                    "transcript": it["transcript"],
                }
            )

    print(f"SUCCESS: 200 clips (100 EN + 100 FR) ready! Metadata saved to {METADATA_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
