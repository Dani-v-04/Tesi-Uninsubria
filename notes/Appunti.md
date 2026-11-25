# Appunti

## 1. Introduzione

Questa dispensa raccoglie, in ordine, tutto ciò che ho imparato da zero fino al linear probe su MedSigLIP.

Ogni sezione segue lo stesso schema: l'idea in una frase, un esempio semplice, e dove compare nel codice che ho scritto.

Il percorso, nell'ordine in cui l'ho fatto:

1. **Preparazione** (Colab, Drive, GitHub, token di Hugging Face, Zotero): notebook `01`, primo test zero-shot.
2. **PyTorch di base** (sessioni 1-3): tensori, gradiente, loss, ciclo di addestramento, `Dataset` e `DataLoader`: notebook `02` e `03`.
3. **Primo modello su immagini e generalizzazione** (sessioni 4-5): MNIST, overfitting, validation, early stopping: notebook `04`.
4. **Metriche** (sessione 6): matrice di confusione, precisione, recall, soglia, ROC e AUC: notebook `05`.
5. **Embedding** (sessione 7): la mappa degli animali, distanza e coseno, gli embedding veri di MedSigLIP: notebook `06` e `07`.
6. **Logit, softmax e sigmoide** (sessione 8): cosa significano i numeri del primo test: notebook `07`.
7. **Dataset e linear probe** (sessioni 9-10): PneumoniaMNIST, embedding salvati, classificatore, soglie e test: notebook `08` e `09`.

Il filo che lega tutto: un modello trasforma un'immagine in numeri (embedding), da quei numeri si ottiene un punteggio (logit), il punteggio si trasforma in una decisione con una soglia, e la decisione si misura con le metriche.

## 2. Cos'è MedSigLIP?

MedSigLIP (`google/medsiglip-448`) è un modello di Google che trasforma immagini mediche e frasi nello stesso spazio numerico, così da poterle confrontare.

**Com'è fatto.** È una variante di SigLIP-400M con due "torri": una per le immagini (un Vision Transformer da circa 400 milioni di parametri) e una per il testo (un transformer da circa 400 milioni di parametri). In tutto circa 0,9 miliardi di parametri. Le immagini sono 448×448 pixel con valori normalizzati tra −1 e 1, il testo arriva a 64 token.

**Cosa produce.** Per ogni immagine e per ogni frase un embedding (nel nostro caso 1152 numeri), e un punteggio di somiglianza tra immagine e frase.

**A cosa serve** (secondo la model card):

- **Classificazione zero-shot**: si descrivono le classi con frasi e si sceglie quella più vicina all'immagine, senza addestrare niente.
- **Classificazione con pochi dati** (linear probe): si calcolano gli embedding e sopra si allena un classificatore semplice.
- **Recupero semantico**: trovare immagini simili a una descrizione.

Per generare testo (ad esempio un referto) esiste un altro modello della stessa famiglia, MedGemma; MedSigLIP ne è l'encoder per le immagini.

MedSigLIP non è un dispositivo medico e non va usato per decisioni cliniche. Per scaricarlo servono un account Hugging Face e l'accettazione dei termini di Health AI Developer Foundations.

## 3. PyTorch di base

Un modello impara cambiando i suoi parametri a piccoli passi, nella direzione che fa diminuire l'errore.

**Tensore.** È una tabella di numeri con una forma (`.shape`): `[50]` è una lista di 50 numeri, `[50, 1]` una colonna di 50, `[6, 2]` sei righe da 2 numeri. Un'immagine MNIST è `[1, 28, 28]` (1 canale, 28×28 pixel); una di MedSigLIP è 3 canali, 448×448. Operazioni utili: `unsqueeze(1)` aggiunge una dimensione (da `[50]` a `[50, 1]`), `flatten` schiaccia in una fila, `.item()` estrae un numero, `.detach()` stacca dal calcolo del gradiente.

**Modello.** Nella retta della sessione 1 il modello è una funzione con due parametri, w (pendenza) e b (intercetta):

```math
\hat{y} = w \cdot x + b
```

I dati finti erano y ≈ 2x + 1 più un po' di rumore. L'addestramento cerca i valori di w e b che avvicinano la retta ai dati.

**Loss (funzione di errore).** Misura quanto il modello sbaglia, con un solo numero. Qui è l'errore quadratico medio (MSE):

```math
L = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
```

Si eleva al quadrato perché gli errori positivi e negativi non si annullino e perché quelli grandi pesino di più. Anche con i valori giusti (w = 2, b = 1) la loss non va a zero: resta circa 0,01, il quadrato del rumore (0,1²). Nessun modello può prevedere un disturbo casuale.

**Autograd e gradiente.** Il gradiente dice come cambia la loss se si muove ogni parametro. ES: scendere da una collina nella nebbia sentendo l'inclinazione del terreno. Con `requires_grad=True` PyTorch prende appunti sulle operazioni; con `loss.backward()` li rilegge all'indietro e mette le derivate in `.grad`. Gradiente negativo vuol dire: aumentando quel parametro la loss scende.

**Discesa del gradiente.** Il passo di aggiornamento è:

```math
\theta \leftarrow \theta - \eta \, \nabla L
```

θ è un parametro, η (`lr`, learning rate) la lunghezza del passo, ∇L il gradiente. Il segno meno non è una convenzione: indica la direzione in cui la loss scende. Con il segno sbagliato la loss sale (nei miei test: da 2,40 a 3,04).

**Learning rate.** Troppo piccolo: si impara lentamente. Troppo grande: si scavalca la valle e la loss peggiora (con `lr = 5` la loss è esplosa a 89). Il valore giusto dipende dal problema: su MNIST `lr = 1.0` funzionava, sulla retta no.

**Ciclo di addestramento.** Si ripetono, per molti passi:

1. azzerare i gradienti (`optimizer.zero_grad()`)
2. calcolare la previsione e la loss
3. `loss.backward()`: calcolare i gradienti
4. `optimizer.step()`: aggiornare i parametri

L'azzeramento serve perché PyTorch **somma** i nuovi gradienti a quelli vecchi: senza azzerarli i valori oscillano e non si stabilizzano (nel mio esperimento w arrivava a circa 4 e poi tornava vicino a 0). Più passi non significano sempre più precisione: dopo un certo punto la loss si appiattisce, perché il gradiente è quasi zero.

**Gli strumenti di PyTorch.** Fanno lo stesso lavoro con meno scrittura: `nn.Linear(1, 1)` contiene w e b (`model.weight`, `model.bias`) e vuole input in colonna `[N, 1]`; `nn.MSELoss()` calcola l'MSE; `torch.optim.SGD(model.parameters(), lr=0.1)` aggiorna tutti i parametri insieme. Un solo passo con questi strumenti dà gli stessi numeri del passo fatto a mano (loss da circa 2,40 a circa 1,85).

Con milioni di parametri non si possono creare a mano: servono contenitori (`nn.Module`) e un optimizer. MedSigLIP è costruito con mattoncini di questo tipo.

## 4. Dati e addestramento su immagini: Dataset, batch, GPU, MNIST

Per addestrare un modello su molti dati si danno gli esempi a gruppetti (batch), si ripete il ciclo per più epoche e si usa la GPU per andare veloci.

**Dataset.** È lo schedario: sa quanti esempi ci sono (`len`) e restituisce l'esempio numero i. `TensorDataset` abbina tensori già in memoria, riga per riga, quindi x e y restano uniti.

**DataLoader.** Consegna gli esempi a gruppetti. Con `shuffle=True` li rimescola a ogni epoca, ma le coppie (x, y) restano unite. L'ultimo gruppo può essere più piccolo (50 esempi con `batch_size=16` danno 3 gruppi da 16 e uno da 2).

**Epoca e batch.** Un'epoca è un passaggio completo su tutti i dati. Con 50 esempi e `batch_size=10` ogni epoca ha 5 aggiornamenti. La loss media dell'epoca si calcola pesando ogni gruppo per la sua dimensione (`loss.item() * len(xb)`, diviso per il totale).

**Effetto di `batch_size`.** Come `lr`: lo scelgo io, non lo impara il modello.

- Gruppi piccoli: più aggiornamenti per epoca, quindi si impara prima, ma ogni passo è rumoroso e la GPU è sfruttata male.
- Gruppi grandi: passi più stabili ed efficienti, ma meno aggiornamenti per epoca e più memoria.
- Valori tipici: da 16 a 256. Con MedSigLIP su Colab, per limiti di memoria, si usano gruppi piccoli (8-16).

**GPU.** Il modello si sposta una volta con `model.to(device)`; i tensori vanno riassegnati a ogni gruppo (`xb = xb.to(device)`). Con `device: cpu` in Colab è attiva la CPU: si abilita la GPU da Runtime, Cambia tipo di runtime.

Un'immagine MNIST ha 28×28 = 784 pixel con valori tra 0 e 1. `nn.Flatten` la schiaccia in una fila di 784 numeri, `nn.Sequential` mette i pezzi in fila.

**Parametri.** Il modello lineare di MNIST ha 784 pesi × 10 classi + 10 bias = 7850 parametri. Il 784 dipende dai dati, il 10 dal problema (le cifre), il totale da come si costruisce il modello.

**Logit.** Sono i punteggi grezzi che il modello dà a ogni classe prima di trasformarli in probabilità: possono essere positivi, negativi, grandi o piccoli. In MNIST sono 10, uno per cifra, e la cifra scelta è quella con il punteggio più alto (`argmax`).

**Softmax.** Trasforma i logit in quote che sommano sempre 100%:

```math
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
```

**Cross-entropy (`nn.CrossEntropyLoss`).** Riceve i logit e l'etichetta intera, calcola le probabilità con la softmax e punisce una bassa probabilità sulla classe giusta. Per un modello che tira a indovinare tra 10 classi vale circa ln(10) = 2,30.

**Valutazione.** Si usa `torch.no_grad()` e non servono `zero_grad`, `backward` né `step`. L'accuratezza è la frazione di previsioni giuste (`argmax` confrontato con l'etichetta).

**Risultati su MNIST (modello lineare, 3 epoche, `batch_size=64`).**

| `lr` | loss epoca 0 | loss epoca 1 | loss epoca 2 |
| --- | --- | --- | --- |
| 0,01 | 0,98 | 0,55 | 0,47 |
| 0,1 | 0,48 | 0,34 | 0,31 |
| 1,0 | 0,42 | 0,33 | 0,32 |

Con `lr = 0.1` l'accuratezza finale è 0,916 sul train e 0,918 sul test. Prima dell'addestramento era circa 0,12, il livello del caso.

**Classi equilibrate o no.** In MNIST le cifre sono circa 6000 ciascuna (verificato con `bincount`). Nei dataset medici spesso non lo sono, e l'accuratezza da sola può ingannare (sezione 6).

## 5. Generalizzazione: train, validation, test e overfitting

Un modello va giudicato su esempi che non ha usato per imparare, altrimenti si misura la memoria e non la capacità di generalizzare.

- **Train**: il libro di esercizi con le soluzioni. Il modello ci studia.
- **Validation** (val): la simulazione d'esame. Serve a prendere decisioni (quando fermarsi, quale `lr`, quale soglia), ma non conta come voto.
- **Test**: l'esame vero. Si guarda una sola volta, alla fine, senza rieseguire per migliorarlo: altrimenti anche il test diventa un insieme su cui si è "studiato".

**Overfitting.** Il modello memorizza invece di imparare la regola generale. Il segnale non è un'accuratezza alta (con 60.000 esempi il train era al 92% e andava bene), ma questo: la misura sui dati non usati per imparare smette di migliorare mentre quella sul train continua a salire.

Esperimento: stesso modello lineare su soli 500 esempi di MNIST, 200 epoche, `lr = 0.1`. L'accuratezza sul train sale da 0,77 a 1,00, quella sul test da 0,66 a 0,83 e si appiattisce dopo circa 40-60 epoche: un divario finale di circa 17 punti. Con tutti i 60.000 esempi il divario era nullo (train 0,916, test 0,918).

**Errore da evitare.** Fermarsi perché "l'accuratezza è troppo alta" non è il criterio giusto. Alla 20ª epoca il test saliva ancora.

**Early stopping.** Si ferma l'addestramento quando la misura sul validation non migliora per un certo numero di epoche consecutive (la "pazienza") e si ripristinano i pesi migliori (`copy.deepcopy(model.state_dict())` per salvarli, `load_state_dict` per ripristinarli). Nel mio esperimento (validation di 1000 immagini): senza early stopping il massimo era 0,804 all'epoca 48; con pazienza 20 l'addestramento si è fermato all'epoca 51, con il miglior validation a 0,806 (epoca 31). Il test è 0,826, praticamente uguale a quello dopo 200 epoche (0,834), ma ottenuto con circa un quarto delle epoche.

**Rumore statistico.** Con pochi esempi l'accuratezza oscilla per puro caso. L'incertezza è circa ±4 punti con 100 immagini, ±1,3 con 1000, ±0,4 con 10.000. Differenze piccole non sono significative, e il massimo di una curva rumorosa è sempre ottimistico.

**Più dati o più epoche.** Più dati contano più di più epoche (500 esempi: test 0,834; 60.000 esempi: 0,918). Con dataset medici piccoli l'overfitting è il rischio principale, ed è uno dei motivi per usare modelli pre-addestrati come MedSigLIP.

**Iperparametri.** Sono le scelte che fa chi addestra, non il modello: `lr`, `batch_size`, numero di epoche, pazienza, la soglia di decisione. Si scelgono guardando il validation.

## 6. Metriche

Con classi sbilanciate l'accuratezza inganna, e il tipo di errore che costa di più decide quale metrica conta.

|  | il test dice "malato" | il test dice "sano" |
| --- | --- | --- |
| è malato | vero positivo (TP) | falso negativo (FN): malato non scoperto |
| è sano | falso positivo (FP): falso allarme | vero negativo (TN) |

L'insieme dei quattro numeri è la **matrice di confusione**. Nella matrice di `scikit-learn` le righe sono la realtà (prima i sani, poi i malati) e le colonne la risposta del modello.

**Le misure.**

```math
\begin{gathered} \text{accuratezza} = \frac{TP + TN}{TP + TN + FP + FN} \\ \text{precisione} = \frac{TP}{TP + FP} \\ \text{recall} = \frac{TP}{TP + FN} \end{gathered}
```

```math
\begin{gathered} \text{specificità} = \frac{TN}{TN + FP} \\ F_1 = \frac{2 \cdot \text{precisione} \cdot \text{recall}}{\text{precisione} + \text{recall}} \end{gathered}
```

- **Precisione**: quando il modello dice "malato", quante volte ha ragione.
- **Recall** (sensibilità): dei malati veri, quanti ne trova.
- **Specificità**: dei sani, quanti riconosce come sani.
- **F1**: una media di precisione e recall che penalizza chi è scarso su una delle due.

**Esempio.** 100 pazienti, 10 malati e 90 sani. Il modello trova 6 malati su 10 (TP = 6, FN = 4) e dà falso allarme per 9 sani (FP = 9, TN = 81). Matrice `[[81, 9], [4, 6]]`: accuratezza 0,87, precisione 0,40, recall 0,60, specificità 0,90, F1 0,48.

**Perché l'accuratezza inganna.** Un "modello" che dice sempre "sano" ha su quel campione accuratezza 0,90 (TN = 90), più alta del modello vero, ma recall 0 e precisione indefinita (0/0: in `scikit-learn` si usa `zero_division=0`). Il valore "gratis" di partenza è la frazione della classe più numerosa.

**Quale errore costa di più.** Dipende dall'uso.

- **Screening** (cercare malati in una popolazione): l'errore grave è il falso negativo, quindi serve recall alta. I falsi allarmi si controllano con un secondo esame.
- **Conferma prima di un intervento invasivo**: l'errore grave è il falso positivo (operare un sano), quindi serve precisione alta.
- La recall da sola non basta: il modello "sempre malato" ha recall 1,00 ma specificità 0. Si leggono insieme.

**Punteggio e soglia.** Un modello vero produce un punteggio (un logit); la soglia lo trasforma in risposta: "malato" se il punteggio è almeno la soglia. Abbassando la soglia sale la recall ma salgono anche i falsi allarmi: è un compromesso, non c'è un valore migliore in assoluto. Con punteggi sovrapposti (un malato con punteggio più basso di un sano) nessuna soglia dà recall 1 e zero falsi allarmi. Una soglia che a parità di specificità ha recall più bassa di un'altra è semplicemente peggiore.

**Curva ROC.** Si mette ogni soglia come un punto: falsi allarmi (1 − specificità) sull'asse x, recall sull'asse y. Non servono soglie a passi regolari: contano solo quelle che coincidono con i punteggi, perché la risposta cambia solo quando la soglia supera il punteggio di qualcuno. Scendendo con la soglia, la curva sale se incontra un malato e va a destra se incontra un sano. Una curva vicina all'angolo in alto a sinistra è un buon modello; la diagonale è il caso. `roc_curve` salta i punti in mezzo ai tratti dritti.

**AUC.** È l'area sotto la curva ROC, un numero tra 0,5 (caso) e 1,0 (separazione perfetta). Ha un significato concreto: la probabilità che, scegliendo un malato a caso e un sano a caso, il malato abbia il punteggio più alto. Esempio a mano con 4 malati (0,9; 0,8; 0,6; 0,4) e 6 sani (0,7; 0,5; 0,3; 0,2; 0,2; 0,1): su 4 × 6 = 24 coppie il malato vince in 21, quindi AUC = 21/24 = 0,875.

**Perché l'AUC serve a confrontare modelli.** Dipende solo dall'ordine dei punteggi, non dalla loro scala né dalla soglia. Riscalando i punteggi senza cambiare l'ordine (nuovo = 0,5 + vecchio/2) recall e falsi allarmi a soglia 0,5 cambiano (da 0,75 e 0,33 a 1,00 e 1,00), ma l'AUC resta 0,875. Per questo la model card di MedSigLIP riporta AUC.

**Come leggere "AUC 0,844".** Prendendo una radiografia con la patologia e una senza, nell'84,4% dei casi il modello dà il punteggio più alto a quella con la patologia. Nella model card è una media di più AUC, una per patologia.

## 7. Embedding

Un embedding è la riga di numeri che rappresenta un oggetto (un'immagine, una frase), costruita in modo che oggetti simili abbiano righe vicine.

**Immagine di riferimento.** Una mappa: una città si descrive con due numeri (latitudine, longitudine) e città vicine hanno coordinate vicine. Un embedding è una mappa in cui le "città" sono immagini o frasi e "vicino" significa "simile di significato". Il modello non usa 2 numeri ma centinaia: MedSigLIP ne usa 1152.

**Tabella di embedding.** Una riga per oggetto, una colonna per ogni numero. Gli animali della sessione 7 formavano una tabella `[6, 2]` (6 animali, 2 numeri: grandezza e selvatichezza). L'embedding di una immagine di MedSigLIP è `[1, 1152]`, quello di 4 frasi `[4, 1152]`.

**L'ordine delle colonne è fisso.** Ogni posizione ha sempre lo stesso significato per tutti gli oggetti: scambiare le colonne descrive un altro oggetto. Esempio: il lupo (5, 10) in (grandezza, selvatichezza) è un animale medio e molto selvatico, vicino alla tigre (distanza circa 3,16); l'elefante è (10, 5), l'opposto, a distanza circa 7,07.

**Colonne nei modelli veri.** Non le sceglie chi costruisce il modello: le impara il modello guardando moltissimi esempi, e di solito non hanno un nome leggibile.

**Distanza.** La distanza di Pitagora tra due embedding:

```math
d(a, b) = \sqrt{(a_1 - b_1)^2 + (a_2 - b_2)^2 + \dots}
```

Numero piccolo = oggetti simili. Nella mappa degli animali: gatto-topo ≈ 1,41, gatto-cane = 2, gatto-leone ≈ 9,90.

**Similarità coseno.** Confronta la direzione di due vettori e ignora quanto sono "lunghi":

```math
\cos(a, b) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert}
```

Valori: 1 stessa direzione, 0 direzioni perpendicolari (nessuna relazione), −1 direzioni opposte.

**Perché il coseno e non la distanza.** Esempio con tre vettori: a = (2, 2), b = (4, 4), c = (4, 1). Distanza a-b = 2,83 e a-c = 2,24; coseno a-b = 1,00 e a-c = 0,86. Le due misure danno risposte opposte. L'immagine è una ricetta: a è 2 uova e 2 etti di farina, b è la stessa ricetta in versione doppia (stesse proporzioni), c è un'altra ricetta. La distanza vede b come lontana; il coseno la vede uguale. Il significato di un embedding sta nelle proporzioni tra i numeri (la direzione); la lunghezza può dipendere da fattori poco importanti (contrasto o luminosità dell'immagine, scala interna del modello).

**Normalizzazione.** Dividere un vettore per la sua lunghezza lo porta a lunghezza 1:

```math
\hat{v} = \frac{v}{\lVert v \rVert}
```

Per vettori di lunghezza 1 il coseno coincide con il prodotto scalare (si moltiplicano i numeri a coppie e si sommano). Esempio: (0,6; 0,8) e (0,8; 0,6) hanno lunghezza 1 e prodotto scalare 0,96. MedSigLIP normalizza gli embedding di immagini e frasi: nel mio test le lunghezze erano 1,0000.

## 8. MedSigLIP in pratica: dal coseno alle percentuali

Il punteggio che MedSigLIP dà a una coppia immagine-frase nasce dal coseno dei due embedding, moltiplicato per una scala e sommato a un bias.

**Zero-shot.** Si descrivono le classi con frasi ("a photo of an arm with a rash") e si sceglie la frase il cui embedding è più vicino a quello dell'immagine, senza addestrare niente.

**Come si usa il modello** (notebook `01` e `07`):

- `model = AutoModel.from_pretrained("google/medsiglip-448").to(device)` e `processor = AutoProcessor.from_pretrained(...)`: scarica modello e preparatore di immagini e testo.
- `inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt").to(device)`: prepara immagine e frasi (le frasi vanno riempite fino alla lunghezza massima).
- `outputs = model(**inputs)` dentro `with torch.no_grad()`: è una scatola con più cose: `image_embeds`, `text_embeds`, `logits_per_image`.

**La catena completa.**

1. immagine → embedding (1152 numeri, lunghezza 1); frasi → embedding (`[4, 1152]`)
2. coseno tra l'immagine e ogni frase (`img @ txt.T`, forma `[1, 4]`)
3. logit = coseno × scala + bias
4. softmax o sigmoide → percentuali

```math
\text{logit} = \cos(\text{immagine}, \text{frase}) \cdot s + b
```

con s = `model.logit_scale.exp()` e b = `model.logit_bias`, due numeri imparati dal modello.

**I numeri del primo test** (immagine di esempio del dataset SCIN, 445×153 pixel):

| frase | coseno | logit | softmax (4 frasi) |
| --- | --- | --- | --- |
| braccio senza irritazione | 0,0499 | −9,5023 | 20,61% |
| braccio con irritazione | 0,1281 | −8,7178 | 45,16% |
| gamba senza irritazione | −0,0020 | −10,0231 | 12,24% |
| gamba con irritazione | 0,0564 | −9,4374 | 21,99% |

La scala (10,0296) è il numero per cui moltiplico i coseni per allargarne le differenze; il bias (−10,0029) è un numero che sommo dopo. Il punteggio (logit) di ogni frase è quindi coseno × scala + bias. I logit calcolati a mano sono identici a `logits_per_image`. L'ordine dei coseni coincide con quello delle percentuali: moltiplicare per un numero positivo, sommare una costante e applicare la softmax non cambiano l'ordine.

**Ruolo della scala.** I coseni sono piccoli e vicini tra loro (da −0,002 a 0,128). La softmax guarda le differenze tra i logit: con scala 1 darebbe circa 25% a ciascuna frase ("non so scegliere"); con scala circa 10 le differenze si allargano e prende posizione.

**Ruolo del bias.** È sommato uguale a tutti i logit, quindi la softmax lo ignora. Conta quando i punteggi si leggono uno per uno, come con la sigmoide.

**Softmax e sigmoide.**

- **Softmax**: le frasi competono, le quote sommano sempre 100%. Risponde a "quale è la migliore tra queste?". Dà sempre un vincitore, anche se nessuna frase è adatta, e il valore dipende dalle frasi concorrenti.
- **Sigmoide**: ogni logit è trasformato da solo in un valore tra 0 e 1. Risponde a "questa frase descrive questa immagine, sì o no?". Il valore di una frase non dipende dalle altre.

```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```

Esempio con logit 2, 1, 0: softmax 66,5%, 24,5%, 9,0%; sigmoide 88,1%, 73,1%, 50,0%. Con logit −5, 0, +5 la sigmoide vale 0,7%, 50%, 99,3%.

**Esperimento di controllo** (6 frasi, una sola immagine):

| frase | softmax | sigmoide |
| --- | --- | --- |
| braccio senza irritazione | 14,87% | 0,0075% |
| braccio con irritazione | 32,59% | 0,0164% |
| gamba senza irritazione | 8,83% | 0,0044% |
| gamba con irritazione | 15,87% | 0,0080% |
| a photo of an arm | 21,72% | 0,0109% |
| a photo of a banana | 6,12% | 0,0031% |

Osservazioni:

- Aggiungendo frasi la softmax dei primi quattro valori cambia (da 45,16% a 32,59% per braccio con irritazione), la sigmoide no.
- I valori assoluti della sigmoide sono minuscoli per tutte le frasi, anche per "an arm" (0,0109%): in questo checkpoint e con questa immagine non sono interpretabili come "probabilità che la frase sia giusta". Conta l'ordine.
- Con logit molto negativi softmax e sigmoide sono proporzionali: il rapporto softmax/sigmoide è circa costante (tra 1974 e 2007).
- La banana è la più bassa in entrambe, ma la softmax le dà comunque 6,12%, e "gamba senza irritazione" è già molto vicina in sigmoide.

## 9. Linear probe su PneumoniaMNIST: risultati e come leggerli

Su radiografie pediatriche (normale contro polmonite) un classificatore lineare sopra gli embedding di MedSigLIP raggiunge AUC 0,9925 sul test, ma la soglia scelta sul validation non si trasferisce perfettamente.

1. Procurarsi un dataset con etichette e dividerlo in train, val e test.
2. Calcolare **una volta sola** l'embedding di MedSigLIP di ogni immagine e salvarlo.
3. Allenare sopra gli embedding un classificatore semplice, misurarlo con le metriche della sezione 6.
4. Confrontarlo con lo zero-shot.

**Il dataset.** PneumoniaMNIST fa parte di MedMNIST: radiografie del torace di bambini, 0 = normale, 1 = polmonite, scaricato a 224×224 (`size=224`) e già diviso in tre insiemi. Serve solo per fare pratica, non ha valore clinico. Il processor ingrandisce le immagini a 448×448, quindi non sono alla risoluzione nativa di MedSigLIP.

| insieme | normali | polmonite | frazione polmonite |
| --- | --- | --- | --- |
| train | 1214 | 3494 | 0,742 |
| val | 135 | 389 | 0,742 |
| test | 234 | 390 | 0,625 |

Il modello "sempre polmonite" avrebbe accuratezza 0,742 su train e val e 0,625 su test: è il valore "gratis" da superare.

**Gli embedding.** La funzione `embed` prende le immagini a gruppetti, le prepara con il processor,  e normalizza a lunghezza 1. Controllo: la differenza massima rispetto a `image_embeds` del metodo già noto è 5,3 × 10⁻⁷. Calcolo: 8 immagini in circa 2,4 secondi; val in 2,9 minuti; test e train in 29,5 minuti. Salvati su Drive a blocchi di 500 (`{insieme}_{inizio}.pt`) con le etichette (`{insieme}_labels.npy`): se Colab si disconnette si riparte dal primo blocco mancante.

**Il classificatore.** È una somma pesata della scheda di 1152 numeri più un bias:

```math
\text{punteggio} = w_1 x_1 + w_2 x_2 + \dots + w_{1152} x_{1152} + b
```

Sono 1152 pesi + 1 bias = 1153 parametri. Il punteggio è un logit: sopra la soglia (di solito 0) il modello dice "polmonite". Si è usata `LogisticRegression(max_iter=1000)` di `scikit-learn`, con impostazioni predefinite. Funziona bene perché MedSigLIP ha già separato le classi nello spazio degli embedding.

Corrispondenza con MNIST:

| cosa | in MNIST | nel linear probe |
| --- | --- | --- |
| i dati | immagini `xb`, etichette `yb` | schede `Xtr` e etichette `ytr` |
| il modello | `nn.Linear(784, 10)` | `clf = LogisticRegression(...)` |
| l'addestramento | il ciclo con `zero_grad`, loss, `backward`, `step` | `clf.fit(Xtr.numpy(), ytr)` |
| i punteggi | `model(xb)` | `clf.decision_function(X)` |
| la decisione | `argmax` | `clf.predict(X)` o `punteggio >= soglia` |

I pesi e il bias si vedono in `clf.coef_` (forma `(1, 1152)`) e `clf.intercept_` (forma `(1,)`). Un punteggio si ricostruisce a mano con `x @ clf.coef_[0] + clf.intercept_[0]` (per la prima radiografia del val: 7,2415, identico a `decision_function`). I singoli pesi non sono "giusti" o "sbagliati": ogni peso moltiplica una colonna senza nome leggibile; il modello si giudica dalle prestazioni.

**Risultati.**

| insieme | AUC |
| --- | --- |
| train | 0,9972 |
| val | 0,9967 |
| test | 0,9925 |

Train e val quasi uguali: nessun segno di overfitting (1153 parametri contro 4708 esempi).

Validation, soglia 0: matrice `[[128, 7], [6, 383]]`, recall 0,985, specificità 0,948, precisione 0,982, accuratezza 0,975 (contro 0,742 del modello "sempre polmonite").

**Soglie sul validation.**

| soglia | recall | specificità | FN | FP |
| --- | --- | --- | --- | --- |
| −3 | 1,000 | 0,267 | 0 | 99 |
| −2 | 1,000 | 0,563 | 0 | 59 |
| −1 | 0,997 | 0,837 | 1 | 22 |
| 0 | 0,985 | 0,948 | 6 | 7 |
| 1 | 0,941 | 0,993 | 23 | 1 |
| 2 | 0,895 | 0,993 | 41 | 1 |

Abbassando la soglia sale la recall e scende la specificità. La soglia 2 è peggiore della 1 (stessa specificità, recall più bassa). Per uno screening una soglia attorno a −1 era una candidata: perde 1 malato su 389 al prezzo di 22 falsi allarmi.

**Test**, valutato una sola volta con due soglie fissate prima di guardarlo (234 normali, 390 polmoniti):

| soglia | recall | specificità | FN | FP | accuratezza |
| --- | --- | --- | --- | --- | --- |
| 0 | 0,992 | 0,863 | 3 | 32 | 0,944 |
| −1 | 1,000 | 0,705 | 0 | 69 | 0,889 |

Riferimento "sempre polmonite" sul test: 0,625.

- Il modello generalizza: AUC 0,9925 su dati mai visti.
- Con la stessa soglia 0 la specificità scende da 0,948 (val) a 0,863 (test), più del rumore statistico (circa ±2 punti con 135 normali), mentre la recall resta alta e l'AUC cala poco. Con la soglia −1 i falsi allarmi passano da circa il 16% dei sani (val) a circa il 30% (test).
- L'AUC misura solo l'ordine, che resta buono; dove cade la linea che separa le classi (la soglia) può spostarsi quando cambiano i dati. Il motivo probabile: il test è una parte diversa del dataset originale (il suo "validation" originale), con meno polmoniti (62,5% contro 74,2%), mentre train e val sono ricavati dal training set originale diviso 9:1. Che sia proprio questa la causa non l'ho verificato.
- In tesi conviene riportare AUC e anche i risultati a soglie fissate, e avvertire che la soglia va ricalibrata sui dati del contesto d'uso.

**Rilevatore di fumo.** Se lo rendo molto sensibile non mi sfugge nessun incendio, ma suona anche per il tostapane; se lo rendo poco sensibile non suona a vuoto, ma rischio di perdere un incendio vero.

## 10. Funzioni importanti che mi devo ricordare

**Tensori e calcolo del gradiente (PyTorch)**

| istruzione | cosa fa | dove |
| --- | --- | --- |
| `torch.tensor([...])` | crea un tensore da una lista | sessione 1, animali |
| `x.shape` | dimensioni del tensore (righe, colonne, ...) | ovunque |
| `x.unsqueeze(1)` | aggiunge una dimensione: `[50]` diventa `[50, 1]` | sessione 2 |
| `x.item()` | estrae un numero normale da un tensore di un solo elemento | loss, scala, bias |
| `x.detach()` | stacca un tensore dal calcolo del gradiente | sessione 1 |
| `requires_grad=True` | chiede a PyTorch di tenere appunti per il gradiente | sessione 1 |
| `loss.backward()` | calcola i gradienti e li mette in `.grad` | ciclo di addestramento |
| `w.grad.zero_()` | azzera il gradiente di un parametro | sessione 1 |
| `with torch.no_grad():` | spegne gli appunti del gradiente (valutazione, embedding) | MNIST, embedding |
| `x.to(device)` | sposta un tensore sulla GPU (o CPU) | MNIST, MedSigLIP |
| `x.cpu()` | riporta un tensore dalla GPU alla memoria normale | funzione `embed` |
| `x.numpy()` | converte un tensore in array NumPy (per `scikit-learn`) | linear probe |

**Costruire e addestrare un modello**

| istruzione | cosa fa | dove |
| --- | --- | --- |
| `nn.Linear(a, b)` | strato lineare: `a` numeri in ingresso, `b` in uscita (a·b pesi + b bias) | sessioni 2 e 4 |
| `nn.Flatten()` | schiaccia ogni immagine in una fila di numeri | MNIST |
| `nn.Sequential(...)` | mette i pezzi in fila | MNIST |
| `nn.MSELoss()` | errore quadratico medio | sessione 2 |
| `nn.CrossEntropyLoss()` | softmax + errore sulla classe giusta, per la classificazione | MNIST |
| `torch.optim.SGD(model.parameters(), lr=...)` | aggiorna tutti i parametri con la discesa del gradiente | sessioni 2-5 |
| `optimizer.zero_grad()` | azzera i gradienti a inizio passo | ciclo di addestramento |
| `optimizer.step()` | aggiorna i parametri | ciclo di addestramento |
| `TensorDataset(X, y)` | abbina dati ed etichette riga per riga | sessione 3 |
| `DataLoader(ds, batch_size=..., shuffle=True)` | consegna gli esempi a gruppetti, rimescolati | sessioni 3-4 |
| `copy.deepcopy(model.state_dict())` | salva una copia dei pesi (early stopping) | sessione 5 |
| `model.load_state_dict(...)` | ripristina i pesi salvati | sessione 5 |

**Punteggi, probabilità e confronto di embedding**

| istruzione | cosa fa | dove |
| --- | --- | --- |
| `scores.argmax(dim=1)` | indice del punteggio più alto per ogni esempio | MNIST |
| `torch.softmax(x, dim=-1)` | trasforma i logit in quote che sommano 1 | test MedSigLIP |
| `torch.sigmoid(x)` | trasforma ogni logit in un valore tra 0 e 1 | sessione 8 |
| `v.norm(dim=-1)` | lunghezza di ogni riga | sessione 7 |
| `v / v.norm(dim=-1, keepdim=True)` | porta ogni riga a lunghezza 1 | funzione `embed` |
| `img @ txt.T` | prodotto scalare tra ogni immagine e ogni frase (con vettori di lunghezza 1 è il coseno) | sessione 7 |
| `torch.dist(a, b)` | distanza di Pitagora | sessione 7 |
| `F.cosine_similarity(a, b, dim=0)` | coseno tra due vettori | sessione 7 |
| `torch.cat(lista)` | incolla tensori uno dopo l'altro | funzione `embed` |
| `torch.save(x, path)` e `torch.load(path)` | scrive e rilegge un tensore da file | embedding su Drive |

**MedSigLIP (Hugging Face)**

| istruzione | cosa fa | dove |
| --- | --- | --- |
| `login(token=userdata.get("HF_TOKEN"))` | accede a Hugging Face con il token nei Secrets di Colab | cella di setup |
| `AutoModel.from_pretrained("google/medsiglip-448")` | scarica il modello | notebook `01`, `07`, `09` |
| `AutoProcessor.from_pretrained(...)` | scarica il preparatore di immagini e testo | idem |
| `processor(text=..., images=..., padding="max_length", return_tensors="pt")` | prepara le frasi (lunghezza fissa) e le immagini (448×448, normalizzate) | test zero-shot |
| `model(**inputs)` | esegue il modello; restituisce una scatola (`outputs`) | test zero-shot |
| `outputs.image_embeds`, `outputs.text_embeds` | embedding normalizzati di immagine e frasi | sessione 7 |
| `outputs.logits_per_image` | punteggi (logit) immagine-frasi | test zero-shot |
| `model.logit_scale.exp()`, `model.logit_bias` | scala e bias del modello | sessione 7 |
| `model.get_image_features(**inputs).pooler_output` | embedding delle sole immagini (non normalizzato) | funzione `embed` |

**Metriche e classificatore (scikit-learn e NumPy)**

| istruzione | cosa fa | dove |
| --- | --- | --- |
| `confusion_matrix(y, pred)` | matrice di confusione (righe = realtà) | sessione 6 |
| `confusion_matrix(...).ravel()` | appiattisce la matrice in TN, FP, FN, TP | sessione 6 |
| `precision_score`, `recall_score`, `f1_score`, `accuracy_score` | le metriche della sezione 6 | sessione 6 |
| `roc_curve(y, scores)` | punti (falsi allarmi, recall) per ogni soglia utile | sessione 6 |
| `roc_auc_score(y, scores)` | AUC | sessioni 6 e 10 |
| `LogisticRegression(max_iter=1000)` | classificatore lineare | sessione 10 |
| `clf.fit(X, y)` | addestra (l'intero ciclo di addestramento in una riga) | sessione 10 |
| `clf.decision_function(X)` | logit di ogni esempio | sessione 10 |
| `clf.predict(X)` | risposta con soglia 0 | sessione 10 |
| `clf.coef_`, `clf.intercept_` | pesi e bias del classificatore | sessione 10 |
| `np.bincount(y)` | quanti esempi per classe | MNIST, PneumoniaMNIST |
| `np.save(path, a)` e `np.load(path)` | scrive e rilegge un array NumPy | etichette su Drive |

**Python che serve leggere**

| istruzione | cosa fa | dove |
| --- | --- | --- |
| `a[:, 0]` | tutte le righe, colonna 0 | grafico degli animali |
| `lista[i:i + n]` | un pezzo di lista (da `i`, `n` elementi) | funzione `embed` |
| `range(0, n, passo)` | conta da 0 a `n` saltando di `passo` | funzione `embed` |
| `for a, b in zip(x, y):` | scorre due liste insieme | grafici, stampe |
| `for nome, (x, y) in zip(...)` | scorre e "spacchetta" una coppia | grafico degli animali |
| `[f(i) for i in range(8)]` | costruisce una lista con un ciclo in una riga | sessione 9 |
| `f"{p:.2%}"` | stampa un numero in percentuale con 2 decimali | test zero-shot |
| `(scores >= t).astype(int)` | vero/falso per ogni punteggio, convertito in 1/0 | soglie |
| `os.path.exists(path)` | controlla se un file esiste | salvataggio a blocchi |

**Colab**

| istruzione | cosa fa | dove |
| --- | --- | --- |
| `drive.mount("/content/drive")` | collega Google Drive | cella di setup |
| `userdata.get("HF_TOKEN")` | legge il token dai Secrets (mai nel codice) | cella di setup |
| `!pip install -q pacchetto` | installa una libreria (il `!` esegue un comando del terminale) | `medmnist` |
| "Riavvia ed esegui tutto" | rieseguire il notebook da capo per verificare che sia coerente | tutti i notebook |

## 11. Fonti e metodo di studio

Le formule delle metriche (precisione, recall, specificità, F1, AUC), i valori tipici di iperparametri (per esempio `batch_size` tra 16 e 256, pazienza dell'early stopping tra 5 e 20) e le regole su train, validation e test sono conoscenza standard del machine learning: le trovo in qualsiasi testo introduttivo/tutorial online, ma non le ho prese da uno in particolare.

**Fonti di riferimento per ogni argomento**

- PyTorch (tensori, autograd, `nn.Linear`, `DataLoader`, ciclo di addestramento): tutorial ufficiali su pytorch.org/tutorials; Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep Learning Library*, NeurIPS 2019.
- MNIST: LeCun, Cortes, Burges, *The MNIST database of handwritten digits*.
- Metriche (matrice di confusione, precisione, recall, ROC, AUC) e regressione logistica: documentazione di scikit-learn (guida "Model evaluation" e classe `LogisticRegression`).
- CLIP (immagine e testo nello stesso spazio): Radford et al., arXiv:2103.00020, 2021.
- SigLIP (loss sigmoidea, scala e bias): Zhai et al., *Sigmoid Loss for Language Image Pre-Training*, ICCV 2023, arXiv:2303.15343.
- MedSigLIP e MedGemma: Sellergren et al., *MedGemma Technical Report*, arXiv:2507.05201, 2025; model card su huggingface.co/google/medsiglip-448.
- PneumoniaMNIST: Yang et al., *MedMNIST v2*, arXiv:2110.14795; descrizione del dataset nel catalogo di TensorFlow Datasets.
- Uso di MedSigLIP come estrattore di embedding con un classificatore sopra: Li et al., *From Embeddings to Accuracy*, 2025 (DOI 10.1007/s10278-025-01747-5).

## 12. Glossario

I termini che compaiono nei paper sono quasi sempre in inglese: la colonna centrale mi permette di riconoscerli.

| termine | in inglese | significato |
| --- | --- | --- |
| Accuratezza | accuracy | frazione di risposte giuste; inganna con classi sbilanciate |
| AUC | area under the curve | area sotto la curva ROC: probabilità che un malato a caso abbia punteggio più alto di un sano a caso |
| Autograd | autograd | meccanismo di PyTorch che calcola i gradienti all'indietro |
| Bias | bias | numero sommato al punteggio; nel modello lineare l'intercetta b |
| Classificatore lineare | linear classifier | modello che calcola una somma pesata dei numeri in ingresso |
| Coseno (similarità) | cosine similarity | confronto della direzione di due vettori, da −1 a 1 |
| Cross-entropy | cross-entropy | errore usato in classificazione: punisce una bassa probabilità sulla classe giusta |
| Dataset | dataset | raccolta di esempi (con etichette); in PyTorch lo "schedario" |
| Early stopping | early stopping | fermare l'addestramento quando il validation smette di migliorare |
| Embedding | embedding | riga di numeri che rappresenta un oggetto; oggetti simili hanno righe vicine |
| Epoca | epoch | un passaggio completo su tutti i dati di addestramento |
| F1 | F1 score | media di precisione e recall che penalizza chi è scarso su una delle due |
| Falso negativo (FN) | false negative | malato che il modello dice sano |
| Falso positivo (FP) | false positive | sano che il modello dice malato (falso allarme) |
| GPU | GPU | scheda grafica usata per i calcoli veloci dei modelli |
| Gradiente | gradient | come cambia la loss se si muove ogni parametro |
| Iperparametro | hyperparameter | scelta di chi addestra (`lr`, `batch_size`, soglia), non imparata dal modello |
| Learning rate (`lr`) | learning rate | lunghezza del passo di aggiornamento |
| Linear probe | linear probe | classificatore lineare allenato sopra gli embedding di un modello fisso |
| Logit | logit | punteggio grezzo prima di softmax o sigmoide |
| Loss | loss | numero che misura quanto il modello sbaglia |
| Matrice di confusione | confusion matrix | tabella di TP, FP, FN, TN |
| MedSigLIP | MedSigLIP | variante medica di SigLIP: due torri (immagine e testo) che producono embedding |
| MNIST | MNIST | dataset di cifre scritte a mano (60.000 di train, 10.000 di test) |
| Normalizzazione | normalization | portare un vettore a lunghezza 1 dividendolo per la sua lunghezza |
| Overfitting | overfitting | il modello memorizza il train e non generalizza |
| Parametro | parameter | numero che il modello impara (pesi e bias) |
| Precisione | precision | quando dice "malato", quante volte ha ragione |
| Prodotto scalare | dot product | moltiplicare due vettori a coppie e sommare; con vettori di lunghezza 1 è il coseno |
| Recall (sensibilità) | recall, sensitivity | dei malati veri, quanti ne trova |
| ROC | ROC curve | curva che mostra recall e falsi allarmi per ogni soglia |
| Scala (logit scale) | logit scale | numero imparato che moltiplica il coseno e allarga le differenze |
| Sigmoide | sigmoid | trasforma un logit in un valore tra 0 e 1, indipendente dagli altri |
| Soglia | threshold | valore del punteggio sopra il quale si dice "malato" |
| Softmax | softmax | trasforma i logit in quote che sommano 100% |
| Specificità | specificity | dei sani, quanti riconosce come sani |
| Tensore | tensor | tabella di numeri con una forma |
| Test | test set | insieme per l'esame finale, da guardare una sola volta |
| Train | training set | insieme su cui il modello impara |
| Validation | validation set | insieme per le decisioni (soglia, `lr`, quando fermarsi) |
| Zero-shot | zero-shot | classificare con frasi, senza addestrare sul compito |