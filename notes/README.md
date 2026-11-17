# Diario degli esperimenti – Tesi MedSigLIP

Convenzioni: voci dalla più recente alla più vecchia. Nelle frasi i decimali hanno la virgola; i valori di codice stanno tra apici (`lr = 0.1`). `TODO:` indica qualcosa da completare: si cerca con Ctrl+F.

## Stato del progetto (aggiornato al 2026-11-17)

- **Ambiente:** repository privato `Tesi-Uninsubria` su GitHub; cartelle `tesi-medsiglip/data` e `tesi-medsiglip/results` su Google Drive; esecuzione su Google Colab con GPU. Token di Hugging Face nei Secrets di Colab (mai nel codice).
- **Notebook:**
  - `01_primo_test_medsiglip.ipynb`: primo test zero-shot di MedSigLIP
  - `02_pytorch_basi.ipynb`: autograd, loss, ciclo di addestramento, `nn.Linear`, `MSELoss`, optimizer
  - `03_dataset_dataloader.ipynb`: `Dataset`, `DataLoader`, batch
  - `04_mnist.ipynb`: classificazione di cifre MNIST, overfitting, validation, early stopping
- **Letteratura:** collezione Zotero `tesi` con 5 articoli (MedGemma Technical Report, SigLIP, CLIP, ELIXR e un lavoro che usa MedSigLIP). TODO: tabella degli articoli (titolo, anno, task, dataset, come usano MedSigLIP, risultati).
- **Domande al professore:** dataset da usare e obiettivo concreto del progetto. TODO: annotare la risposta.
- **Da registrare:** esperimenti di comprensione su MedSigLIP (softmax contro sigmoide, frasi riformulate, frase assurda). TODO.
- **Prossimi passi:** metriche di classificazione (precision, recall, F1, AUC), poi embedding di MedSigLIP e classificatore sopra gli embedding.

---

## 2026-11-17 – PyTorch, sessione 5: overfitting, validation e early stopping

**Obiettivo:** capire come si riconosce l'overfitting e come evitare di usare il test per prendere decisioni.

**Cosa ho fatto:** in `04_mnist.ipynb` ho confrontato l'accuratezza su train e test, ho addestrato lo stesso modello lineare su soli 500 esempi per 200 epoche, ho costruito un validation set (immagini 500-1499 del train) e ho implementato l'early stopping con pazienza 20.

**Risultati:**
- Modello su tutti i 60.000 esempi, 3 epoche: train 0,916, test 0,918, divario nullo.
- Modello su 500 esempi, 200 epoche (`lr = 0.1`): train da 0,77 a 1,00, test da 0,66 a 0,83. Il test si appiattisce dopo circa 40-60 epoche. Divario finale di circa 17 punti.
- Validation (1000 immagini): curva piatta dopo circa 25 epoche, massimo 0,804 all'epoca 48, valore finale 0,794. Differenza dentro il rumore.
- Early stopping (pazienza 20): fermo all'epoca 51, miglior validation 0,806 (epoca 31), test 0,826. Praticamente uguale al test dopo 200 epoche (0,834), ma con circa un quarto delle epoche.

**Cosa ho capito:**
- Overfitting = il modello memorizza invece di imparare. Il segnale è che la misura su dati non usati per imparare smette di migliorare mentre quella sul train continua a salire. Un'accuratezza alta da sola non basta come segnale (con 60.000 esempi il train era al 92% e andava bene).
- Errore di ragionamento corretto: avevo scelto di fermarmi all'epoca 20 perché "un'accuratezza troppo alta vuol dire memorizzazione". Il criterio giusto è il plateau della misura su dati nuovi: alla 20 il test saliva ancora, il plateau era intorno a 40-60.
- Servono tre insiemi: train (impara), validation (decisioni: quando fermarsi, quale `lr`), test (si guarda una volta sola alla fine, senza rieseguire per migliorarlo).
- Il massimo di una curva rumorosa è ottimistico. L'incertezza dell'accuratezza dipende dal numero di esempi: circa ±4 punti con 100 immagini, ±1,3 con 1000, ±0,4 con 10.000. Differenze piccole non sono significative.
- Più dati contano più di più epoche (500 esempi: test 0,834; 60.000 esempi: 0,918). Con dataset medici piccoli l'overfitting è il rischio principale, ed è uno dei motivi per usare modelli pre-addestrati come MedSigLIP.
- Un notebook ricorda lo stato: la prima misura era 0,12 (livello del caso) perché il modello era stato ricreato senza rieseguire l'addestramento. Verificare con "Riavvia ed esegui tutto".

**Domande aperte:** TODO

**Prossimo passo:** metriche per la classificazione (precision, recall, F1, AUC) e classificatore sugli embedding.

---

## 2026-11-17 – PyTorch, sessione 4: classificazione di cifre con MNIST

**Obiettivo:** addestrare una rete su immagini vere e misurarla su dati mai visti, con lo stesso schema che userò con MedSigLIP.

**Cosa ho fatto:** in `04_mnist.ipynb` ho scaricato MNIST (60.000 immagini di train, 10.000 di test), costruito un modello lineare 784→10, l'ho addestrato su GPU per 3 epoche (`CrossEntropyLoss`, SGD, `batch_size = 64`) e l'ho valutato sul test.

**Idee chiave:**
- Un'immagine è un tensore di numeri: in MNIST `[1, 28, 28]` (1 canale, 28×28 pixel, valori tra 0 e 1), cioè 784 numeri. In MedSigLIP sarà 3 canali, 448×448, valori tra -1 e 1.
- Il modello trasforma i 784 pixel in 10 punteggi (logit), uno per cifra; la cifra scelta è quella col punteggio più alto (`argmax`).
- Parametri: 784 pesi × 10 classi + 10 bias = 7850. Il 784 dipende dai dati (28×28), il 10 dal problema (10 cifre), il totale da come costruisco il modello (qui un solo strato lineare).
- `nn.Flatten` schiaccia ogni immagine in una fila di 784 numeri; `nn.Sequential` mette i pezzi in fila.
- `CrossEntropyLoss`: riceve i logit e l'etichetta intera, calcola le probabilità (softmax) e punisce una bassa probabilità sulla cifra giusta. Per un modello che tira a indovinare vale circa ln(10) = 2,30.
- Il ciclo è quello della sessione 3, più lo spostamento dei dati sulla GPU: il modello si sposta una volta con `model.to(device)`, i tensori vanno riassegnati (`xb = xb.to(device)`).
- Nella valutazione si usa `torch.no_grad()` e non servono `zero_grad`, `backward` né `step`.
- Il test set non viene mai usato per addestrare.
- Le classi di MNIST sono equilibrate (circa 6000 immagini per cifra, verificato con `bincount`). Nei dataset medici spesso non lo sono, quindi l'accuratezza da sola può ingannare e si usano metriche come l'AUC.

**Risultati (3 epoche, loss media dell'epoca):**
- `lr = 0.01`: 0,98, 0,55, 0,47 (ancora in discesa)
- `lr = 0.1`: 0,48, 0,34, 0,31
- `lr = 1.0`: 0,42, 0,33, 0,32
- Prima dell'addestramento: loss circa 2,30 e accuratezza circa 10% (misurata 0,12 con modello non addestrato).
- Dopo l'addestramento (`lr = 0.1`): train 0,916, test 0,918.

**Cosa mi ha sorpreso:** `lr = 1.0` funziona su questo problema, mentre sulla retta `lr = 5` esplodeva: il valore massimo tollerato dipende dal problema. Alla terza epoca `lr = 0.1` è leggermente sotto `lr = 1.0`, anche se all'inizio è più lento; la differenza è piccola e viene da una sola esecuzione, quindi non è una conclusione (servirebbe ripetere con semi diversi).

**Errori del modello:** TODO (cosa ho visto nelle immagini sbagliate, per esempio quali cifre confonde)

**Domande aperte:** TODO

**Prossimo passo:** sessione 5 (overfitting e validation).

---

## 2026-11-15 – PyTorch, sessione 3: Dataset, DataLoader e batch

**Obiettivo:** capire come si danno i dati al modello a gruppetti (batch) e mettere insieme `DataLoader` e ciclo di addestramento.

**Cosa ho fatto:** in `03_dataset_dataloader.ipynb` ho creato un `TensorDataset` con i 50 punti, un `DataLoader` con `batch_size = 10` e `shuffle = True`, e ho scritto il ciclo con due `for` annidati (epoche e gruppetti) usando `nn.Linear`, `MSELoss` e SGD (`lr = 0.1`).

**Idee chiave:**
- `Dataset` è lo schedario: sa quanti esempi ci sono (`len`) e dà l'esempio numero i. `TensorDataset` abbina tensori già in memoria, riga per riga (x e y restano insieme).
- `DataLoader` consegna gli esempi a gruppetti. Con `shuffle = True` li rimescola a ogni epoca, ma le coppie (x, y) restano unite. L'ultimo gruppo può essere più piccolo (50 esempi con `batch_size = 16` danno 3 gruppi da 16 e uno da 2).
- Un'epoca è un passaggio completo su tutti i dati. Con 50 esempi e `batch_size = 10` ogni epoca ha 5 aggiornamenti.
- Struttura del ciclo: per ogni epoca, per ogni gruppo `(xb, yb)`: `zero_grad`, previsione, loss, `backward`, `step`.
- La loss media dell'epoca si calcola pesando ogni gruppo per la sua dimensione: `loss.item() * len(xb)`, poi divisa per il totale degli esempi.

**Risultati (`batch_size = 10`, `lr = 0.1`, 20 epoche, misurati nel mio notebook):**
- epoca 0: loss 1,608, w 0,606, b 0,693
- epoca 4: loss 0,0748, w 1,694, b 1,009
- epoca 19: loss 0,0115, w 2,027, b 1,004

**Esperimenti sul `batch_size`:**
- Gruppi più grandi: meno aggiornamenti per epoca, quindi si impara più lentamente per epoca. Gruppi più piccoli: già alla prima epoca i valori sono più vicini a quelli giusti.
- TODO: annotare i numeri misurati con `batch_size = 50` e `batch_size = 5` (20 epoche) e verificare con 100 epoche che anche `batch_size = 50` arriva allo stesso punto.

**Cosa ho capito:**
- Gruppi piccoli: più aggiornamenti per epoca, quindi si impara prima, ma ogni passo è rumoroso (pochi esempi danno un gradiente poco affidabile) e la GPU è sfruttata male.
- Gruppi grandi: passi più stabili ed efficienti, ma meno aggiornamenti per epoca e più memoria. Con `batch_size = 1` si impara in fretta per epoca ma con un cammino a zigzag.
- `batch_size` è un iperparametro, come `lr`: lo scelgo io, non lo impara il modello. Valori tipici tra 16 e 256. Con MedSigLIP su Colab dovrò probabilmente usare gruppi piccoli (8-16) per limiti di memoria.

**Domande aperte:** TODO

**Prossimo passo:** addestrare una vera rete su immagini (MNIST) con la GPU.

---

## 2026-11-13 – PyTorch, sessione 2: `nn.Linear`, `MSELoss` e optimizer

**Obiettivo:** sostituire i pezzi scritti a mano (parametri, loss, aggiornamento) con gli strumenti di PyTorch e verificare che diano gli stessi risultati.

**Cosa ho fatto:** in `02_pytorch_basi.ipynb` ho creato un modello con `nn.Linear(1, 1)`, ho usato `nn.MSELoss` e `torch.optim.SGD` (`lr = 0.1`) ed ho eseguito un solo passo di addestramento partendo da w = 0 e b = 0.

**Idee chiave:**
- Il modello è una retta: y = w·x + b. L'ho scelta io perché i dati stavano su una retta. w è la pendenza, b l'intercetta; l'addestramento cerca i valori che avvicinano la retta ai dati.
- `nn.Linear(1, 1)` è la stessa retta, con w e b conservati dentro l'oggetto (`model.weight` e `model.bias`). `model(Xc)` significa "dai `Xc` al modello e restituisci w·x + b per ogni x".
- `nn.Linear` vuole gli input in colonna: forma `[50, 1]` e non `[50]`. Per questo uso `X.unsqueeze(1)`. Il contenuto non cambia, cambia solo la disposizione.
- `nn.MSELoss()` è l'errore quadratico medio, uguale a `((pred - Y)**2).mean()`.
- `optimizer.step()` aggiorna i parametri (nuovo = vecchio − `lr` × gradiente), `optimizer.zero_grad()` azzera i gradienti. Si mette all'inizio del passo.

**Risultati (un passo, `lr = 0.1`):** loss prima ≈ 2,40; dopo il passo w ≈ 0,14 e b ≈ 0,2; loss dopo ≈ 1,85. Gli stessi numeri del passo fatto a mano: il modello e i calcoli sono identici, cambia solo il modo di scriverli.

**Cosa ho capito:** con molti parametri non posso creare ogni numero a mano: servono contenitori (`nn.Module`) e un optimizer che li aggiorni tutti insieme. MedSigLIP è costruito con mattoncini di questo tipo.

**Prossimo passo:** mettere queste righe dentro un `for` per ottenere il ciclo di addestramento completo.

---

## 2026-11-11 – PyTorch, sessione 1: autograd, gradiente, passo di discesa e ciclo

**Obiettivo:** capire come un modello "impara" prima di usare modelli veri.

**Cosa ho fatto:** in `02_pytorch_basi.ipynb` ho usato autograd su y = x² + 2x; ho creato dati finti con y ≈ 2x + 1 più rumore; ho costruito un modello con due parametri (w e b); ho calcolato loss e gradienti; ho fatto un solo aggiornamento a mano variando `lr` e segno; ho scritto il ciclo completo di 100 passi e ho provato due variazioni.

**Idee chiave:**
- Un modello ha dei parametri (qui w e b). La loss misura quanto sbaglia, il gradiente dice come cambia la loss se muovo ogni parametro. Immagine che uso: scendere da una collina nella nebbia sentendo l'inclinazione del terreno.
- Con `requires_grad=True` PyTorch prende appunti sulle operazioni; con `backward()` li rilegge all'indietro e mette le derivate in `.grad`.
- La loss è l'errore quadratico medio: gli errori al quadrato non si annullano tra positivi e negativi e quelli grandi pesano di più.
- Gradiente negativo = aumentando quel parametro la loss scende. Formula del passo: nuovo = vecchio − `lr` × gradiente (i due meno si annullano e il parametro aumenta, che è la direzione giusta).

**Risultati:**
- Partenza (w = 0, b = 0): loss ≈ 2,40, `w.grad` ≈ −1,39, `b.grad` ≈ −2,00. Con i valori veri (w = 2, b = 1): loss ≈ 0,01 e gradienti vicini a zero.
- Un passo, `lr = 0.1`: w ≈ 0,14, b ≈ 0,2, loss ≈ 1,85. Con `lr = 0.5`: w ≈ 0,69, b ≈ 1,0, loss ≈ 0,60. Con `lr = 5`: loss ≈ 89 (esplode). Con il segno + al posto del −: w ≈ −0,14, b ≈ −0,2, loss ≈ 3,04 (sale).
- Ciclo di 100 passi (`lr = 0.1`): passo 0 loss ≈ 2,4; passo 20 loss ≈ 0,09, w ≈ 1,5, b ≈ 1,0; dal passo 90 loss ≈ 0,01, w ≈ 2,0, b ≈ 1,0.
- Con 20 passi w non è ancora a 2 (circa 1,5); con 300 passi non si migliora rispetto a 100.
- Senza `zero_()`: w e b oscillano senza stabilizzarsi (w arriva a circa 4, poi torna vicino a 0), la loss a 100 passi resta a circa 0,7.

**Cosa ho capito:**
- Il gradiente dice in che direzione scendere, non quanto è lontano il fondo. Un `lr` troppo piccolo rende l'apprendimento lento; troppo grande fa scavalcare la valle e la loss peggiora. Il valore giusto dipende dal problema.
- Più passi non significano sempre più precisione: oltre un certo punto la loss si appiattisce (fondo della valle, gradiente quasi zero).
- `zero_()` serve perché PyTorch somma i nuovi gradienti a quelli precedenti: senza azzerarli ogni passo usa la somma di tutti i precedenti, come uno slancio cieco. Il momentum è una tecnica voluta e controllata, diversa da questo accumulo.
- Il segno meno nella formula non è una convenzione: indica la direzione in cui la loss scende.

**Cosa mi ha sorpreso:** anche con i valori veri la loss non va a zero: è il rumore (0,1² = 0,01), nessun modello può prevedere un disturbo casuale. Per lo stesso motivo w finale ≈ 2,06 e non 2: la retta migliore sui dati rumorosi non coincide con quella vera. Se la loss andasse a zero sui dati di addestramento sarebbe sospetto (overfitting).

**Domande aperte:** TODO

**Prossimo passo:** `nn.Linear` e gli optimizer.

---

## 2026-11-03 – Primo test zero-shot di MedSigLIP su Colab

**Obiettivo:** verificare che il modello giri nell'ambiente (Colab + GPU + Drive).

**Cosa ho fatto:** ho caricato `google/medsiglip-448`, un'immagine di esempio della model card e 4 frasi candidate.

**Risultato:** braccio con irritazione 45,2%, gamba con irritazione 22,0%, braccio senza 20,6%, gamba senza 12,2%.

**Da fare:** TODO confronto softmax e sigmoide, frasi riformulate, frase assurda, poi embedding.

---

## 2026-11-03 – Setup ambiente

**Obiettivo:** preparare repository GitHub, cartelle su Drive e Colab.

**Cosa ho fatto:** creato il repository privato con `.gitignore`, struttura di cartelle su Git e su Drive, README, `requirements.txt`, Zotero con la collezione `tesi`.

**Risultato:** struttura pronta; notebook su Colab con GPU funzionante.

**Domande aperte:** quale dataset userà il professore per il progetto? TODO annotare la risposta.