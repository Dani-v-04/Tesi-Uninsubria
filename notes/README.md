## 2026-11-13 – PyTorch, sessione 2: nn.Linear, MSELoss e optimizer

**Obiettivo:** sostituire i pezzi scritti a mano (parametri, loss, aggiornamento) con gli strumenti di PyTorch e verificare che diano gli stessi risultati
**Cosa ho fatto:** nel notebook 02_pytorch_basi.ipynb ho creato un modello con nn.Linear(1, 1), ho usato nn.MSELoss e torch.optim.SGD (lr = 0.1) ed ho eseguito un solo passo di addestramento partendo da w = 0 e b = 0.

**Idee chiave:**
- Il modello è una retta: y = w·x + b. Lo abbiamo scelto noi perché i dati stavano su una retta. w è la pendenza, b l'intercetta, l'addestramento cerca i valori che avvicinano la retta ai dati.
- nn.Linear(1, 1) è la stessa retta, con w e b conservati dentro l'oggetto (model.weight e model.bias). model(Xc) significa "dai Xc al modello e restituisci w·x + b per ogni x".
- nn.Linear vuole gli input in colonna: forma [50, 1] e non [50]. Per questo uso X.unsqueeze(1). Il contenuto non cambia, cambia solo la disposizione.
- nn.MSELoss() è l'errore quadratico medio, uguale a ((pred - Y)**2).mean().
- optimizer.step() aggiorna i parametri (parametro nuovo = vecchio − lr × gradiente), optimizer.zero_grad() azzera i gradienti. Si mette all'inizio del passo.

**Risultati (un passo, lr = 0.1):**
- loss prima del passo: ≈ 2.40
- dopo il passo: w ≈ 0.14, b ≈ 0.2
- loss dopo il passo: ≈ 1.85
Gli stessi numeri del passo fatto a mano: il modello e i calcoli sono identici, cambia solo il modo di scriverli.

**Cosa ho capito:** con molti parametri non posso creare ogni numero a mano: servono contenitori (nn.Module) e un optimizer che li aggiorni tutti insieme. MedSigLIP è costruito con mattoncini di questo tipo.

**Prossimo passo:** mettere queste righe dentro un for per ottenere il ciclo di addestramento completo

## 2026-11-11 – PyTorch, sessione 1 (passo 4): il ciclo di addestramento

**Obiettivo:** ripetere il passo di apprendimento in un ciclo e capire il ruolo di numero di passi e azzeramento dei gradienti
**Cosa ho fatto:** nel notebook 02_pytorch_basi.ipynb ho messo le quattro mosse (loss, backward, aggiornamento, azzeramento) in un ciclo for con lr = 0.1 e le ho ripetute 100 volte, poi ho provato due variazioni.

**Risultati (ciclo corretto, lr = 0.1):**
- passo 0: loss ≈ 2.4, w = 0, b = 0
- passo 20: loss ≈ 0.09, w ≈ 1.5, b ≈ 1.0
- passo 90 e oltre: loss ≈ 0.01, w ≈ 2.0, b ≈ 1.0

**Esperimento 1 – numero di passi:**
- Con 20 passi w non è ancora arrivato a 2 (circa 1.5), b invece sì.
- Con 300 passi non si migliora rispetto a 100: dopo circa 100 passi la loss si ferma, perché sono in fondo alla valle e il gradiente è quasi zero.
- w finale ≈ 2.06 e non 2: la retta migliore sui dati rumorosi non coincide con quella vera.

**Esperimento 2 – senza zero_():**
- I valori di w e b oscillano senza stabilizzarsi (w arriva fino a circa 4, poi torna vicino a 0). A 100 passi la loss resta a circa 0.7, contro 0.01 del ciclo corretto.
- Motivo: PyTorch somma il nuovo gradiente a quello che c'era già, quindi ogni passo usa la somma di tutti i gradienti precedenti. È come avere uno slancio che mi fa superare il fondo della valle.

**Cosa ho capito:**
- Più passi non significa sempre più precisione: oltre un certo punto la loss si appiattisce.
- zero_() serve perché ogni passo deve basarsi solo sul gradiente di quel momento.
- Il momentum esiste come tecnica voluta, ma è controllato; qui l'accumulo è cieco.

**Prossimo passo:** sessione 2: nn.Linear e gli optimizer, che fanno al posto mio quasi tutte queste righe

## 2026-11-11 – PyTorch, sessione 1 (passo 3): un solo aggiornamento a mano

**Obiettivo:** capire cosa fa un singolo passo di apprendimento prima di scrivere il ciclo completo
**Cosa ho fatto:** nel notebook 02_pytorch_basi.ipynb ho eseguito un solo aggiornamento di w e b partendo da w = 0, b = 0, con learning rate (lr) = 0.1, e poi ho ripetuto il passo cambiando lr e il segno.

**Formula:** parametro nuovo = parametro vecchio − lr × gradiente.
Il gradiente di w è negativo (circa −1.39), quindi i due meno si annullano e w aumenta. È la direzione giusta, perché i punti salgono verso destra.

**Risultati (partenza: loss = 2.40, w.grad ≈ −1.39, b.grad ≈ −2.00):**
- lr = 0.1: w ≈ 0.14, b ≈ 0.2, loss ≈ 1.85 (scende)
- lr = 0.5: w ≈ 0.69, b ≈ 1.0, loss ≈ 0.60 (scende molto di più)
- lr = 0.1 con il segno + al posto del −: w ≈ −0.14, b ≈ −0.2, loss ≈ 3.04 (sale, vado nella direzione sbagliata)
- lr = 5: loss ≈ 89 (esplode)

**Cosa ho capito:**
- Il gradiente dice in che direzione scendere ma non quanto è lontano il fondo.
- Un lr troppo piccolo rende l'apprendimento lento. Un lr troppo grande fa scavalcare la valle e atterrare più in alto di dove ero partito, quindi la loss peggiora a ogni passo.
- Il valore giusto di lr dipende dal problema: 0.5 è andato bene su questi dati, ma su altri potrebbe già essere troppo.
- Il segno meno nella formula non è una convenzione: indica la direzione in cui la loss scende.

**Prossimo passo:** mettere questo stesso gesto dentro un ciclo che lo ripete molte volte

## 2026-11-11 – PyTorch, sessione 1 (passi 1-2): autograd, loss, gradiente

**Obiettivo:** capire come un modello "impara" prima di usare modelli veri
**Cosa ho fatto:** notebook 02_pytorch_basi.ipynb. Ho usato autograd su y = x² + 2x, ho creato dati finti con y ≈ 2x + 1 più rumore, ho costruito un modello con due parametri (w e b) e ho calcolato loss e gradienti.

**Idee chiave:**
- Un modello ha dei parametri (qui w e b). La loss misura quanto sbaglia, il gradiente dice come cambia la loss se muovo ogni parametro. L'immagine che uso: scendere da una collina nella nebbia sentendo l'inclinazione del terreno.
- Con requires_grad=True PyTorch prende appunti sulle operazioni; con backward() li rilegge all'indietro e mette le derivate in .grad.
- La loss è l'errore quadratico medio: si elevano gli errori al quadrato così quelli positivi e negativi non si annullano e quelli grandi pesano di più.
- Gradiente negativo = aumentando quel parametro la loss scende. Per migliorare ci si muove nella direzione opposta al gradiente (per questo ci sarà il meno nell'aggiornamento).

**Risultati:**
- Con w = 0 e b = 0: loss ≈ 2.40, w.grad ≈ −1.39, b.grad ≈ −2.00
- Con i valori veri w = 2 e b = 1: loss ≈ 0.01, gradienti vicini a zero (valori esatti: <copiali dall'output del notebook>)
- Previsione fatta prima di eseguire: w.grad negativo perché w deve aumentare. Confermata.

**Cosa mi ha sorpreso:** anche con i valori veri la loss non va a zero. Il motivo è il rumore (0.1² = 0.01): nessun modello può prevedere un disturbo casuale. Se la loss andasse a zero sui dati di addestramento sarebbe sospetto (overfitting).

**Prossimo passo:** fare un solo aggiornamento a mano di w e b, poi il ciclo completo di addestramento

## 2026-11-03 – Primo test zero-shot di MedSigLIP su Colab

**Obiettivo:** verificare che il modello giri nell'ambiente (Colab + GPU + Drive)
**Cosa ho fatto:** caricato google/medsiglip-448, un'immagine di esempio della model card, 4 frasi candidate
**Risultato:** braccio con irritazione 45,2%, gamba con irritazione 22,0%, braccio senza 20,6%, gamba senza 12,2%
**Prossimo passo:** confronto softmax e sigmoide, frasi riformulate, poi embedding

## 2026-11-03 – Setup ambiente

**Obiettivo:** preparare repository GitHub, cartelle su Drive e Colab
**Cosa ho fatto:** creato repo privato con .gitignore, cartelle su Git e su Drive
**Risultato:** struttura pronta, nessun codice ancora
**Domande aperte:** quale dataset userà il prof per il progetto?
**Prossimo passo:** primo notebook su Colab con MedSigLIP