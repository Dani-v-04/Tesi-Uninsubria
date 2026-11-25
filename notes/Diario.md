# Diario delle attività – Tirocinio su MedSigLIP

Questo diario descrive nel dettaglio le attività svolte, sessione per sessione, dalla più recente alla più vecchia. Il riepilogo sintetico, in forma di tabella, è nel registro delle attività. Il codice è nel repository privato `Tesi-Uninsubria`; dati e risultati sono nella cartella Drive `Tesi-Uninsubria` (`data`, `results`, variabile `BASE` nei notebook). Lavoro su Google Colab con GPU; il token di Hugging Face sta nei Secrets di Colab, mai nel codice. Nelle frasi i decimali hanno la virgola; i valori di codice stanno tra apici (`lr = 0.1`). I numeri riportati sono tutti misurati da me nei notebook, tranne dove indicato.

## Stato al 2026-11-24


## Sessione 10 – 22/11/2026 – Classificatore lineare sugli embedding (linear probe)

**Notebook:** `09_embedding_pneumonia.ipynb`
**Obiettivo:** addestrare un classificatore semplice sopra gli embedding di MedSigLIP e misurarlo con le metriche della sessione 6.

**Attività svolte**
1. Ho caricato dal Drive gli embedding salvati nella sessione 9 (train 4708 immagini, val 524, test 624, ciascuna con 1152 numeri) con la funzione `load_split`, che riunisce i blocchi da 500.
2. Ho addestrato una `LogisticRegression` di scikit-learn (`max_iter = 1000`, impostazioni predefinite) sul train: `clf.fit(Xtr.numpy(), ytr)`. Il modello ha 1152 pesi più 1 bias, cioè 1153 parametri.
3. Ho misurato l'AUC su train, val e test con `decision_function`.
4. Sul val ho calcolato matrice di confusione, recall, specificità, precisione e accuratezza a soglia 0, poi ho provato le soglie −3, −2, −1, 0, 1, 2 (`(s >= t).astype(int)` e `confusion_matrix`).
5. Ho verificato a mano il calcolo del punteggio per la prima radiografia del val: `x @ clf.coef_[0] + clf.intercept_[0]` = 7,2415, identico a `decision_function`.
6. Ho guardato il test una sola volta, con due soglie decise prima sul val (0 e −1).
7. Ho confrontato il classificatore con il caso "sempre polmonite" (accuratezza 0,742 su train e val, 0,625 su test).

**Risultati**
- AUC: train 0,9972, val 0,9967, test 0,9925 (differenza train-val 0,0005: nessun overfitting evidente).
- Val, soglia 0: matrice [[128, 7], [6, 383]]; recall 0,985, specificità 0,948, precisione 0,982, accuratezza 0,975.
- Val, soglie diverse:

| soglia | recall | specificità | falsi negativi | falsi positivi |
|---|---|---|---|---|
| −3 | 1,000 | 0,267 | 0 | 99 |
| −2 | 1,000 | 0,563 | 0 | 59 |
| −1 | 0,997 | 0,837 | 1 | 22 |
| 0 | 0,985 | 0,948 | 6 | 7 |
| 1 | 0,941 | 0,993 | 23 | 1 |
| 2 | 0,895 | 0,993 | 41 | 1 |

- Test (234 normali, 390 polmoniti): soglia 0 → recall 0,992, specificità 0,863, 3 falsi negativi, 32 falsi positivi, accuratezza 0,944; soglia −1 → recall 1,000, specificità 0,705, 0 falsi negativi, 69 falsi positivi, accuratezza 0,889.

**Osservazioni**
- Abbassando la soglia la recall sale e la specificità scende; la soglia 2 è peggiore della 1 (stessa specificità, recall più bassa). Per uno screening ha senso una soglia più bassa, a costo di più falsi allarmi.
- Con soglia 0 la specificità scende da 0,948 (val) a 0,863 (test), più del rumore statistico (circa ±2 punti con 135 normali). La causa probabile è che il test viene da una parte diversa del dataset originale, con meno polmoniti (vedi sessione 9); non l'ho verificato. L'AUC cala poco (0,9967 → 0,9925). Nella tesi riporterò AUC e risultati a soglie fissate, precisando che la soglia va ricalibrata sui dati d'uso.
- Corrispondenza con MNIST: `Xtr`/`ytr` come immagini ed etichette; `LogisticRegression` come `nn.Linear`; `clf.fit` come l'intero ciclo di addestramento; `decision_function` come `model(X)`; `predict` come `argmax`. I singoli pesi non sono giusti o sbagliati: il modello si giudica dalle prestazioni.

---

## Sessione 9 – 20/11/2026 – PneumoniaMNIST ed embedding di tutte le radiografie

**Notebook:** `08_dataset_pneumonia.ipynb`, `09_embedding_pneumonia.ipynb`
**Obiettivo:** procurarmi un dataset medico etichettato e calcolare gli embedding di MedSigLIP per tutte le immagini.

**Attività svolte**
1. In `08` ho scaricato PneumoniaMNIST (libreria `medmnist`, `size=224`, radiografie del torace di bambini, 0 = normale, 1 = polmonite) e ho contato le classi nei tre insiemi.
2. In `09` ho scritto la funzione `embed(images, batch_size=8)`: per ogni gruppetto chiama il `processor` (ingrandisce a 448), poi `model.get_image_features(**inputs).pooler_output` dentro `torch.no_grad()`, poi normalizza a lunghezza 1 e riporta sulla CPU.
3. Ho provato `embed` su 8 immagini: forma `[8, 1152]`, lunghezze tutte 1, 2,44 secondi (secondo giro).
4. Ho verificato la correttezza confrontando il mio embedding con `image_embeds` di MedSigLIP: differenza massima 5,3 × 10⁻⁷.
5. Ho scritto `embed_split(ds, name, chunk=500)`, che salva su Drive i blocchi `{name}_{start}.pt` e le etichette `{name}_labels.npy`, saltando i blocchi già fatti (utile se Colab si disconnette), e `load_split`.
6. Ho calcolato e salvato gli embedding di tutte le 5856 immagini in `Tesi-Uninsubria/results/emb/`: val in 2,9 minuti, test e train insieme in 29,5 minuti.

**Risultati**

| insieme | normali | polmonite | frazione polmonite |
|---|---|---|---|
| train | 1214 | 3494 | 0,742 |
| val | 135 | 389 | 0,742 |
| test | 234 | 390 | 0,625 |

- Un modello che dice sempre "polmonite" avrebbe accuratezza 0,742 su train e val e 0,625 sul test.
- Dalla descrizione ufficiale di PneumoniaMNIST: train e val sono ricavati dividendo 9:1 il training set del dataset di partenza, mentre il test è il "validation" originale. Per questo le proporzioni nel test sono diverse.

**Sintassi nuova usata:** `range(start, stop, passo)`, `lista[i:i + n]`, `torch.cat`, `.cpu()`, `os.path.exists`, `torch.save`/`torch.load`, `np.save`, `!pip install`.

---

## Sessione 8 – 20/11/2026 – Logit, softmax e sigmoide

**Notebook:** `07_embedding_medsiglip.ipynb`
**Obiettivo:** capire cosa sono i logit e cosa significano le percentuali del primo test, confrontando softmax e sigmoide.

**Attività svolte**
1. Ho applicato `torch.sigmoid` ai punteggi (logit) del primo test.
2. Ho aggiunto due frasi di controllo, "a photo of an arm" e "a photo of a banana", e ho confrontato softmax (4 e 6 frasi) e sigmoide sulle sei frasi, sulla stessa immagine.
3. Ho calcolato il rapporto softmax / sigmoide per ogni frase.
4. Ho cercato nel paper di SigLIP (Sezione 3.2) come sono inizializzati temperatura e bias: partono da log 10 e −10, cioè scala 10 e bias −10.

**Risultati (6 frasi, una sola immagine)**

| frase | softmax (4 frasi) | softmax (6 frasi) | sigmoide |
|---|---|---|---|
| braccio senza irritazione | 20,61% | 14,87% | 0,0075% |
| braccio con irritazione | 45,16% | 32,59% | 0,0164% |
| gamba senza irritazione | 12,24% | 8,83% | 0,0044% |
| gamba con irritazione | 21,99% | 15,87% | 0,0080% |
| a photo of an arm | – | 21,72% | 0,0109% |
| a photo of a banana | – | 6,12% | 0,0031% |

- Aggiungendo frasi, la sigmoide dei primi quattro valori non cambia; la softmax sì.
- Il rapporto softmax / sigmoide è quasi costante (tra 1974 e 2007).
- "Banana" è la più bassa in entrambe le colonne, ma la softmax le dà comunque 6,12%.

**Osservazioni**
- Softmax: le frasi competono e le percentuali sommano sempre 100%; la sigmoide valuta ogni frase da sola. Con punteggi molto negativi le due sono proporzionali.
- Con questo modello e questa immagine i valori assoluti della sigmoide sono minuscoli anche per frasi ragionevoli, quindi non si leggono come probabilità: conta solo l'ordine. Per questo la model card usa l'AUC.
- Scala e bias del checkpoint (circa 10,03 e −10,00) sono quasi uguali ai valori iniziali del paper (10 e −10). Deduco che si siano mossi poco; è una mia deduzione, non è scritta nel paper.
- Se in un lavoro leggo "90% alla frase giusta", devo guardare quale funzione ha prodotto il numero e con quali frasi concorrenti.

---

## Sessione 7 – 19-20/11/2026 – Embedding: dagli animali a MedSigLIP

**Notebook:** `06_embedding.ipynb` (19/11), `07_embedding_medsiglip.ipynb` (20/11, copia di `01`)
**Obiettivo:** capire cos'è un embedding e ricostruire a mano la catena che porta dall'immagine alle percentuali del primo test.

**Attività svolte**
1. In `06` ho costruito una mappa a 2 numeri (grandezza, selvatichezza) con sei animali e ho calcolato distanze (Pitagora) e similarità coseno.
2. In `07` ho estratto da `outputs` del primo test gli embedding di immagine e frasi e ne ho controllato forma e lunghezza.
3. Ho calcolato il coseno tra immagine e frasi con `img @ txt.T`.
4. Ho letto scala (`model.logit_scale.exp()`) e bias (`model.logit_bias`) e ho ricostruito a mano i punteggi: logit = coseno × scala + bias.
5. Ho applicato la softmax ai punteggi e ho confrontato con il risultato del primo test.

**Risultati**
- Mappa animali: gatto-cane 2, gatto-topo circa 1,41, gatto-leone circa 9,90; tigre-leone 1, tigre-elefante circa 4,47. Con a = (2, 2), b = (4, 4), c = (4, 1): distanza a-b 2,83 e a-c 2,24, coseno a-b 1,00 e a-c 0,86 (le due misure danno risposte opposte).
- Immagine SCIN (445×153 pixel), 4 frasi: `image_embeds` `[1, 1152]`, frasi `[4, 1152]`, lunghezze 1.
- Coseno con l'immagine (braccio senza irritazione, braccio con, gamba senza, gamba con): 0,0499; 0,1281; −0,0020; 0,0564.
- Scala 10,0296, bias −10,0029. Punteggi a mano: −9,5023; −8,7178; −10,0231; −9,4374, identici a `logits_per_image`. Softmax: 20,61%; 45,16%; 12,24%; 21,99%, uguale al primo test.
- Tutti i punteggi sono molto negativi (tra −8,7 e −10,0); la softmax lo nasconde.

**Osservazioni**
- Un embedding è la riga di numeri che rappresenta un oggetto; oggetti simili hanno righe vicine. L'ordine delle colonne ha un significato fisso. Nei modelli veri le colonne le impara il modello e non hanno un nome leggibile: le 1152 colonne di MedSigLIP contano nell'insieme.
- Il coseno confronta la direzione e ignora la lunghezza. MedSigLIP ha due parti (immagine e testo) che portano immagini e frasi nello stesso spazio; gli embedding sono normalizzati a lunghezza 1, quindi coseno = prodotto scalare.
- La scala allarga le differenze tra i coseni (con scala 1 la softmax darebbe circa 25% a ciascuna frase). Il bias non influisce sulla softmax.

---

## Sessione 6 – 18/11/2026 – Metriche di valutazione

**Notebook:** `05_metriche.ipynb`
**Obiettivo:** capire le metriche che compaiono nella model card di MedSigLIP (AUC, sensibilità, specificità).

**Attività svolte**
1. Su un esempio di 100 pazienti (10 malati, 90 sani) ho calcolato con scikit-learn matrice di confusione, accuratezza, precisione, recall e F1; ho verificato che un modello che dice sempre "sano" ha accuratezza 0,90 ma recall 0.
2. Su 10 pazienti con punteggi inventati (4 malati, 6 sani) ho variato la soglia e calcolato recall e falsi allarmi.
3. Ho tracciato la curva ROC con `roc_curve` e calcolato l'AUC, anche a mano contando le coppie (malato, sano).
4. Ho creato un secondo modello (punteggi riscalati con 0,5 + punteggio / 2, stesso ordine) per verificare che l'AUC non dipende dalla scala.
5. Ho riverificato tutti i numeri rieseguendo i calcoli con scikit-learn: coincidono.

**Risultati**
- 100 pazienti (TP = 6, FN = 4, FP = 9, TN = 81): accuratezza 0,87, precisione 0,40, recall 0,60, specificità 0,90, F1 0,48.
- Soglie sui 10 pazienti: 0,75 → recall 0,50, falsi allarmi 0,00; 0,5 → 0,75 e 0,33; 0,35 → 1,00 e 0,33; 0,15 → 1,00 e 0,83.
- AUC a mano: 21 coppie su 24 con il malato più alto, quindi 21 / 24 = 0,875.
- Modello riscalato: a soglia 0,5 dice "malato" a tutti (recall 1,00, falsi allarmi 1,00), ma l'AUC è identica, 0,875.

**Osservazioni**
- Con classi sbilanciate l'accuratezza inganna. La metrica da privilegiare dipende dall'errore che costa di più: screening → recall; conferma prima di un intervento → precisione.
- Il modello produce un punteggio e la soglia lo trasforma in "malato/sano"; abbassarla alza recall e falsi allarmi. Nella ROC contano solo le soglie che coincidono con i punteggi.
- L'AUC è la probabilità che un malato scelto a caso abbia un punteggio più alto di un sano scelto a caso; dipende solo dall'ordine, quindi serve a confrontare modelli. AUC 0,844 nella model card (zero-shot, media tra le patologie) significa questo, mediato su più patologie.

---

## Sessione 5 – 17/11/2026 – Overfitting, validation, early stopping

**Notebook:** `04_mnist.ipynb`
**Obiettivo:** capire come si riconosce l'overfitting ed evitare di usare il test per le decisioni.

**Attività svolte**
1. Ho confrontato l'accuratezza su train e test del modello lineare addestrato su tutti i 60.000 esempi (3 epoche).
2. Ho addestrato lo stesso modello su soli 500 esempi per 200 epoche (`lr = 0.1`) misurando train e test a ogni epoca.
3. Ho costruito un validation set (immagini 500-1499 del train, 1000 immagini) e ho misurato la curva di validation.
4. Ho implementato l'early stopping con pazienza 20.

**Risultati**
- 60.000 esempi, 3 epoche: train 0,916, test 0,918.
- 500 esempi, 200 epoche: train da 0,77 a 1,00, test da 0,66 a 0,83; il test si appiattisce dopo circa 40-60 epoche; divario finale circa 17 punti.
- Validation: curva piatta dopo circa 25 epoche, massimo 0,804 all'epoca 48, finale 0,794.
- Early stopping: fermo all'epoca 51, miglior validation 0,806 (epoca 31), test 0,826, contro 0,834 dopo 200 epoche (circa un quarto delle epoche).

**Osservazioni**
- Il segnale di overfitting è che la misura sui dati non usati per imparare smette di migliorare mentre quella sul train sale. Un'accuratezza alta da sola non basta (con 60.000 esempi il train era al 92% e andava bene).
- Servono tre insiemi: train, validation (decisioni), test (una volta sola, alla fine).
- Incertezza dell'accuratezza: circa ±4 punti con 100 immagini, ±1,3 con 1000, ±0,4 con 10.000.
- Più dati contano più di più epoche (500 esempi: 0,834; 60.000: 0,918). Con dataset medici piccoli il rischio principale è l'overfitting.

---

## Sessione 4 – 17/11/2026 – Classificazione di cifre con MNIST

**Notebook:** `04_mnist.ipynb`
**Obiettivo:** addestrare una rete su immagini vere e valutarla su dati mai visti, con lo schema che userò con MedSigLIP.

**Attività svolte**
1. Ho scaricato MNIST (60.000 immagini di train, 10.000 di test) e ho verificato il bilanciamento delle classi con `bincount` (circa 6000 per cifra).
2. Ho costruito il modello `nn.Sequential(nn.Flatten(), nn.Linear(784, 10))` (7850 parametri) e l'ho spostato su GPU.
3. Ho addestrato per 3 epoche con `CrossEntropyLoss`, SGD e `batch_size = 64`, provando `lr` = 0.01, 0.1 e 1.0.
4. Ho valutato train e test con `torch.no_grad()` e `argmax`.

**Risultati (loss media per epoca, 3 epoche)**
- `lr = 0.01`: 0,98, 0,55, 0,47. `lr = 0.1`: 0,48, 0,34, 0,31. `lr = 1.0`: 0,42, 0,33, 0,32.
- Prima dell'addestramento: loss circa 2,30 (= ln 10) e accuratezza 0,12.
- Dopo (`lr = 0.1`): train 0,916, test 0,918.

**Osservazioni:** `lr = 1.0` funziona qui mentre sulla retta `lr = 5` esplodeva: il valore massimo tollerato dipende dal problema. La differenza tra `lr = 0.1` e `lr = 1.0` alla terza epoca è piccola e viene da una sola esecuzione.

---

## Sessione 3 – 15/11/2026 – Dataset, DataLoader e batch

**Notebook:** `03_dataset_dataloader.ipynb`
**Obiettivo:** dare i dati al modello a gruppetti e mettere insieme `DataLoader` e ciclo di addestramento.

**Attività svolte**
1. Ho creato un `TensorDataset` con i 50 punti e un `DataLoader` con `batch_size = 10` e `shuffle = True`.
2. Ho scritto il ciclo con due `for` annidati (epoche e gruppetti) con `nn.Linear`, `MSELoss` e SGD (`lr = 0.1`): `zero_grad`, previsione, loss, `backward`, `step`.
3. Ho calcolato la loss media dell'epoca pesando ogni gruppo con `loss.item() * len(xb)`.
4. Ho provato `batch_size` diversi (50 e 5) e ho verificato che con `batch_size = 16` i 50 esempi danno 3 gruppi da 16 e uno da 2.

**Risultati (`batch_size = 10`, 20 epoche)**
- epoca 0: loss 1,608, w 0,606, b 0,693; epoca 4: loss 0,0748, w 1,694, b 1,009; epoca 19: loss 0,0115, w 2,027, b 1,004.
- Con gruppi più grandi si impara più lentamente per epoca; con gruppi più piccoli i valori sono più vicini a quelli giusti già alla prima epoca. I numeri esatti delle prove con `batch_size` 50 e 5 non li ho annotati.

**Osservazioni:** gruppi piccoli = più aggiornamenti ma rumorosi; grandi = passi stabili ma meno aggiornamenti e più memoria. `batch_size` è un iperparametro (tipico 16-256); con MedSigLIP su Colab userò probabilmente 8-16.

---

## Sessione 2 – 13/11/2026 – `nn.Linear`, `MSELoss`, optimizer

**Notebook:** `02_pytorch_basi.ipynb`
**Obiettivo:** sostituire i pezzi scritti a mano con gli strumenti di PyTorch e verificare che diano gli stessi risultati.

**Attività svolte**
1. Ho creato `nn.Linear(1, 1)` (retta y = w·x + b) e ho adattato gli input alla forma `[50, 1]` con `unsqueeze(1)`.
2. Ho usato `nn.MSELoss()` (uguale a `((pred - Y)**2).mean()`) e `torch.optim.SGD` (`lr = 0.1`).
3. Ho eseguito un solo passo (`zero_grad`, previsione, loss, `backward`, `step`) partendo da w = 0 e b = 0.

**Risultato:** loss da ≈ 2,40 a ≈ 1,85; w ≈ 0,14, b ≈ 0,2: identici al passo fatto a mano.
**Osservazione:** con molti parametri servono contenitori (`nn.Module`) e un optimizer che li aggiorni tutti; MedSigLIP è costruito così. Oltre a SGD esistono optimizer come Adam, meno sensibili a `lr`.

---

## Sessione 1 – 11/11/2026 – Autograd, gradiente e ciclo di addestramento

**Notebook:** `02_pytorch_basi.ipynb`
**Obiettivo:** capire come un modello "impara".

**Attività svolte**
1. Ho usato autograd su y = x² + 2x (`requires_grad=True`, `backward()`, `.grad`).
2. Ho generato dati finti con y ≈ 2x + 1 più rumore (0,1) e un modello con due parametri, w e b.
3. Ho calcolato la loss (errore quadratico medio) e i gradienti.
4. Ho fatto un aggiornamento a mano (nuovo = vecchio − `lr` × gradiente) variando `lr` (0,1; 0,5; 5) e il segno.
5. Ho scritto il ciclo completo di 100 passi e ho provato due variazioni: 20 e 300 passi, e senza azzerare i gradienti.

**Risultati**
- Partenza (w = 0, b = 0): loss ≈ 2,40, `w.grad` ≈ −1,39, `b.grad` ≈ −2,00. Con w = 2, b = 1: loss ≈ 0,01.
- Un passo: `lr = 0.1` → w ≈ 0,14, b ≈ 0,2, loss ≈ 1,85; `lr = 0.5` → w ≈ 0,69, b ≈ 1,0, loss ≈ 0,60; `lr = 5` → loss ≈ 89 (esplode); segno + → loss ≈ 3,04 (sale).
- Ciclo (`lr = 0.1`): passo 20 loss ≈ 0,09, w ≈ 1,5; dal passo 90 loss ≈ 0,01, w ≈ 2,0, b ≈ 1,0. Con 300 passi non si migliora.
- Senza `zero_()`: w e b oscillano (w arriva a circa 4 e torna vicino a 0), loss a 100 passi ≈ 0,7.

**Osservazioni:** il gradiente dà la direzione, non la distanza dal fondo; `lr` troppo piccolo è lento, troppo grande scavalca la valle. In un problema nuovo parto da 0,01-0,1 e guardo la curva della loss. PyTorch somma i gradienti, quindi vanno azzerati a ogni passo. La loss non va a zero per via del rumore (0,1² = 0,01); se andasse a zero sui dati di addestramento sarebbe sospetto (overfitting).

---

## Primo test di MedSigLIP – 03/11/2026

**Notebook:** `01_primo_test_medsiglip.ipynb`
**Attività svolte:** ho caricato `google/medsiglip-448`, un'immagine di esempio della model card e 4 frasi candidate.
**Risultato:** braccio con irritazione 45,2%, gamba con irritazione 22,0%, braccio senza irritazione 20,6%, gamba senza irritazione 12,2%. Il modello gira nell'ambiente (Colab + GPU + Drive).
**Seguito:** il significato di queste percentuali (softmax) l'ho studiato nelle sessioni 7 e 8, dove ho provato anche una frase assurda. Frasi riformulate e altre immagini non le ho ancora provate.

---

## Setup dell'ambiente – 03/11/2026

**Attività svolte:** repository privato `Tesi-Uninsubria` con `.gitignore`, struttura di cartelle su Git e su Drive, README, `requirements.txt`; collezione Zotero `tesi`; Colab con GPU.
**Risultato:** ambiente pronto e funzionante.

---

## Ricerca bibliografica

Ho raccolto 8 lavori su MedSigLIP e sui modelli da cui deriva, più CLIP. Li ho riassunti in una tabella nel documento "Studio preliminare di MedSigLIP e fondamenti di apprendimento automatico".

| Lavoro | Anno | Come usa MedSigLIP | Risultato principale | Link |
|---|---|---|---|---|
| MedGemma Technical Report (Sellergren et al.) | 2025 | Lavoro di riferimento del modello | La model card indica di citare questo | [arXiv](https://arxiv.org/abs/2507.05201) |
| SigLIP (Zhai et al.) | 2023 | Modello di partenza | Addestramento immagine-testo con loss sigmoidea | [arXiv](https://arxiv.org/abs/2303.15343) |
| ELIXR (Xu et al.) | 2023 | Termine di confronto | Zero-shot su radiografie: AUC media 0,824 contro 0,844 di MedSigLIP (model card) | [arXiv](https://arxiv.org/abs/2308.01317) |
| Li et al., "From Embeddings to Accuracy" | 2025 | Estrattore di embedding con classificatore sopra | 8842 radiografie, 7 classi: MedSigLIP con MLP mAUC 91,0%; MedImageInsight 93,1%; DenseNet121 addestrata per intero 87,2% | [rivista](https://doi.org/10.1007/s10278-025-01747-5), [arXiv](https://arxiv.org/abs/2505.10823) |
| Demiroglu et al. | 2025 | Parte congelata (non riaddestrata) | Qualità di TC a basso dosaggio, con indicazioni scritte inserite nel modello | [arXiv](https://arxiv.org/abs/2511.12256) |
| Kim e Jeong, LaGuadia | 2026 | "Insegnante" per modelli più piccoli | Guida semantica per modelli di patologia più leggeri | [arXiv](https://arxiv.org/abs/2607.11257) |
| Jabin e Ahmad | 2026 | Estrae feature da vetrini | Nella predizione dell'espressione genica, tra cinque modelli, MedSigLIP va in modo moderato | [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.03.02.709012v1) |
| Model card di MedSigLIP (Google, Hugging Face) | – | Descrizione ufficiale | AUC zero-shot media 0,844 sulle radiografie del torace | [Hugging Face](https://huggingface.co/google/medsiglip-448) |

I risultati sono quelli riportati dagli autori. Il lavoro più vicino al mio è Li et al. (embedding + classificatore sopra, come nelle sessioni 9 e 10).

CLIP (Radford et al., 2021): [arXiv](https://arxiv.org/abs/2103.00020)