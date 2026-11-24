# Registro delle attività – Tirocinio su MedSigLIP

Registro delle attività svolte finora, in ordine cronologico. Per ogni attività indico cosa ho fatto, in quale notebook e il risultato. Il codice è nel repository privato `Tesi-Uninsubria`; dati e risultati sono nella cartella Drive `Tesi-Uninsubria` (`data`, `results`). Lavoro su Google Colab con GPU.

## Registro

| Data | Attività | Notebook | Risultato |
|---|---|---|---|
| 03/11 | Setup: repository GitHub privato (`.gitignore`, README, `requirements.txt`), cartelle su Drive, collezione Zotero `tesi`, Colab con GPU | – | Ambiente pronto |
| 03/11 | Primo test zero-shot di MedSigLIP (`google/medsiglip-448`) con un'immagine di esempio e 4 frasi | `01_primo_test_medsiglip` | Il modello gira; frase più probabile "braccio con irritazione" (45,2%) |
| 11/11 | PyTorch 1: autograd, gradiente, loss, passo di discesa a mano, ciclo di addestramento di 100 passi; prove con diversi `lr` e senza azzeramento dei gradienti | `02_pytorch_basi` | Il ciclo converge a w ≈ 2,0 e b ≈ 1,0; con `lr = 5` la loss esplode |
| 13/11 | PyTorch 2: `nn.Linear`, `MSELoss`, `torch.optim.SGD`; verifica che diano gli stessi numeri del calcolo a mano | `02_pytorch_basi` | Risultati identici al passo fatto a mano (loss 2,40 → 1,85) |
| 15/11 | PyTorch 3: `Dataset`, `DataLoader`, batch, epoche; ciclo con due `for`; prove con `batch_size` diversi | `03_dataset_dataloader` | Con `batch_size = 10`, 20 epoche: loss 0,0115, w 2,027, b 1,004 |
| 17/11 | PyTorch 4: classificazione di cifre MNIST con modello lineare 784→10 su GPU, 3 epoche, confronto di tre `lr` | `04_mnist` | Accuratezza train 0,916, test 0,918 (`lr = 0.1`) |
| 17/11 | PyTorch 5: overfitting (500 esempi, 200 epoche), validation set, early stopping con pazienza 20 | `04_mnist` | Con 500 esempi divario train-test di circa 17 punti; early stopping: test 0,826 con circa un quarto delle epoche |
| 18/11 | Metriche: matrice di confusione, accuratezza, precisione, recall, specificità, F1, soglie, curva ROC, AUC (con scikit-learn, su esempi inventati) | `05_metriche` | AUC di esempio 0,875, invariata cambiando la scala dei punteggi; numeri riverificati con scikit-learn |
| 19/11 | Embedding a mano: mappa di sei animali su 2 numeri, distanza e similarità coseno | `06_embedding` | I due criteri possono dare risposte opposte |
| 20/11 | Embedding veri di MedSigLIP: forma (1152 numeri), normalizzazione, coseno tra immagine e frasi, scala e bias; ricostruzione a mano dei punteggi del primo test | `07_embedding_medsiglip` | Punteggi calcolati a mano identici a quelli del modello; scala 10,0296, bias −10,0029 |
| 20/11 | Confronto softmax e sigmoide su 6 frasi (comprese due frasi di controllo) | `07_embedding_medsiglip` | I valori assoluti della sigmoide sono minuscoli (0,003-0,016%) per tutte le frasi, quindi conta solo l'ordine |
| 20/11 | Dataset PneumoniaMNIST (`size=224`): scarico e conteggio delle classi | `08_dataset_pneumonia` | Train 4708, val 524, test 624 immagini; polmonite 74,2% in train e val, 62,5% nel test |
| 20/11 | Calcolo e salvataggio su Drive degli embedding di MedSigLIP per tutte le 5856 radiografie, con controllo di correttezza | `09_embedding_pneumonia` | Nessuna differenza apprezzabile con `image_embeds` (5,3 × 10⁻⁷); circa 32 minuti di calcolo |
| 22/11 | Linear probe: regressione logistica (scikit-learn) sugli embedding; scelta della soglia sul val; valutazione sul test con soglie fissate prima | `09_embedding_pneumonia` | AUC train 0,9972, val 0,9967, test 0,9925; test con soglia 0: recall 0,992, specificità 0,863 |
| 24/11 | Documentazione: aggiornamento di questo registro e stesura di una dispensa dei concetti studiati (`docs/dispensa_concetti.md`) | – | Due documenti nel repository |

## Ricerca bibliografica

Ho raccolto 8 lavori su MedSigLIP e sui modelli da cui deriva (MedGemma Technical Report, SigLIP, ELIXR, Li et al., Demiroglu et al., Kim e Jeong, Jabin e Ahmad, model card di MedSigLIP) più CLIP, con una tabella riassuntiva nel documento "Studio preliminare di MedSigLIP e fondamenti di apprendimento automatico". In Zotero (collezione `tesi`) ci sono per ora 5 articoli. Il lavoro più vicino a quello che ho fatto è Li et al. (embedding di MedSigLIP con un classificatore sopra).