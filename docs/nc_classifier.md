## Classificatore


### Procedura di Allenamento

Per avviare la procedura di allenamento del feature extractor è possibile utilizzare lo script `train_classifier.py` con le seguenti opzioni:

- `--exp_name`: etichetta assegnata alla corrente procedura di allenamento del classificatore;  per differenziare le etichette assegnate al classificatore da quelle assegnate al feature extractor, si suggerisce di utilizzare il prefizzo `nc_`. 
- NOTA: se è stato precedentemente allenato un feature extractor con un dato `exp_name`, è possibile utilizzare la medesima etichetta anche per il classificatore (a meno del prefisso) per caricare automaticamente i pesi del feature extractor velocizzando notevolmente la procedura di training del classificatore.
- `--ds_root_path`: percorso della directory principale contenete il proprio dataset.
- `--device`: tramite questa opzione è possibile selezionare il device su cui sarà effettuata la procedura di allenamento del classificatore; i possibili valori sono i seguenti: `'cuda'` o `'cuda:<numero specifica gpu>'` per un allenamento su GPU (consigliato) o `'cpu'` per un allenamento su CPU (sconsigliato). Se non specificato, il valore di default è `'cuda'`

Si propone di seguito un esempio di chiamata:

- `python train_classifier.py --exp_name='nc_try1' --ds_root_path='/nas/dataset/nowcast_ds'`
- NOTA: qualora sia stato avviato in precedenza un training del feature extractor con l'opzione `--exp_name=fx_try1`, all'avvio della procedura di training del classificatore, il feature extractor parecaricherà i pesi di `fx_try1`.



Per gli utenti più esperti, è possibile modificare il file `conf.py` per personalizzare i parametri del training; salvo casi molto particolari, tuttavia, si suggerisce di utilizzare i parametri di default. Per completezza, si riporta la porzione del file di configurazione relativa all'allenamento del feature extractor:


```python
# nowcasting classifier settings
NC_LR = 0.0001  # learning rate used to trane the nowcasting classifier
NC_N_WORKERS = 4  # worker(s) number of the dataloader
NC_BATCH_SIZE = 8  # batch size used to trane the nowcasting classifier
NC_MAX_EPOCHS = 256  # maximum training duration (# epochs)
NC_PATIENCE = 16 # stop training if no improvement is seen for a ‘NC_PATIENCE’ number of epochs
```