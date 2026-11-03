## 2026-11-03 – Primo test zero-shot di MedSigLIP su Colab

**Obiettivo:** verificare che il modello giri nell'ambiente (Colab + GPU + Drive)
**Cosa ho fatto:** caricato google/medsiglip-448, un'immagine di esempio della model card, 4 frasi candidate
**Risultato:** braccio con irritazione 45,2%, gamba con irritazione 22,0%, braccio senza 20,6%, gamba senza 12,2%
**Cosa mi ha sorpreso / domande aperte:** (scrivilo tu dopo le celle 6-8)
**Prossimo passo:** confronto softmax e sigmoide, frasi riformulate, poi embedding

## 2026-11-03 – Setup ambiente

**Obiettivo:** preparare repository GitHub, cartelle su Drive e Colab
**Cosa ho fatto:** creato repo privato con .gitignore, cartelle su Git e su Drive
**Risultato:** struttura pronta, nessun codice ancora
**Domande aperte:** quale dataset userà il prof per il progetto?
**Prossimo passo:** primo notebook su Colab con MedSigLIP