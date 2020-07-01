# PyNowCast
### Package Python per Algoritmi e Modelli di Nowcasting

Con il termine “nowcasting” facciamo riferimento all’insieme di tecniche finalizzate alla predizione delle condizioni meteorologiche all’istante di tempo attuale o comunque nell’immediato futuro (generalmente entro un massimo 5/10 minuti) circoscritte a una particolare zona di interesse. Questo concetto si affianca spesso a quello più noto di “forecasting”, che riguarda tuttavia previsioni di accuratezza inferiore, ma relative ad una finestra temporale più ampia, arrivando anche a stime di una settimana in avanti. 

Sebbene ad un primo sguardo nowcasting e forecasting possano apparire molto simili tra loro, le finalità di questi strumenti presentano delle sostanziali differenze. Nel caso del nowcasting, infatti, più che fare previsioni su ciò che accadrà a livello meteorologico, il fine è quello di raccogliere precise statistiche relative a uno o più fenomeni di interesse in una zona circoscritta. 

Il nowcasting è quindi un importante strumento statistico può fungere da supporto nelle analisi sul clima finalizzate a descrivere andamento, intensità e variabilità di fenomeni meteorologici, osservati a diversa scala temporale. Tuttavia, automatizzare il processo di nowcasting risulta essere un’operazione piuttosto complessa che, se affrontata tramite metodologia standard, richiede l’acquisto di strumentazione sofisticata e dai costi piuttosto elevati; per questo motivo, il nowcasting è spesso prerogativa delle stazioni meteo più attrezzate. L’idea è quindi quella di ridurre il più possibile il numero e la complessità dei sensori necessari al raccoglimento dei dati di nowcasting per rendere questa pratica maggiormente accessibile.

Il problema del nowcasting può essere efficacemente affrontato affidandosi a tecniche di Computer Vision e Deep Learning, limitando quindi la richiesta di sensori a semplici telecamere RGB. Questo tipo di approccio risulta particolarmente conveniente in termini di semplicità d’uso e di risorse impiegate, ma presenta una serie di problematiche di progettazione e implementazione che non sono facilmente affrontabili dai non esperti del settore. Per questo motivo nasce l’idea di `PyNowCast`, un package Python che permette di gestire algoritmi e modelli di nowcasting basati su DeepLearning in modo semplice e veloce, occupandosi in modo trasparente di tutti gli aspetti più complessi e macchinosi che caratterizzano questo tipo di tecnologia. Tramite PyCast, quindi, il nowcasting tramite Deep Learning sarà alla portata di tutti.

<br>

### 1. Quick Start
...

<br>

### 2. Dataset
Come prima cosa è necessario organizzare i dati sui quali si vuole allenare il modello di nowcasting secondo il semplice schema mostrato in Figura 1. 
Si avrà pertanto una directory principale, indicata con `<dataset_name>`, con un nome a piacere, che dovrà necessariamente contenere due sotto-directory denominate rispettivamente `train` e `test`, che a loro volta devono contenere le varie sotto-directory contenenti le immagini suddivise per classi. 

<br>

| ![](https://github.com/FabioLanzi/PyNowCast/blob/master/resources/ds_tree.jpg) |
| ------------------------------------------------------------ |
| **Figura 1.** Struttura della directory contenente il dataset. I nomi indicati in verde in figura possono essere scelti a piacere. |

<br>

#### 2.1. Directory di Train e Test
La directory `train`, come il nome suggerisce, è quella preposta a ospitare le immagini di training, ovvero quelle sulle quali sarà allenato il modello di nowcasting; allo stesso modo, la directory `test` conterrà le immagini utilizzate per valutare le prestazioni del suddetto modello. Si noti che, per ottenere una valutazione corretta delle prestazioni del modello, gli insiemi composti dalle immagini di training e dalle immagini di test dovrebbero essere completamente disgiunti, quindi privi di immagini comuni.

Le sotto-directory `train` e `test` devono contenere a loro volta `n` sotto-directory, dove `n` (con `n` maggiore o uguale a 2) è il numero di classi scelte per lo specifico problema di nowcasting che si vuole approcciate con il presente framework. La directory di ogni classe deve contenere esclusivamente le immagini relative a quella specifica classe, affiancate al più da un file `sensors.json` contenente le informazioni relative ad eventuali dati aggiuntivi provenienti da appositi sensori (esempio: sensori di umidità, temperatura, pressione, ...).

Per allenare correttamente il modello di nowcasting, si consiglia di avere un training set bilanciato, quindi con un quantitativo di immagini simile per ogni classe; si consiglia inoltre di avere un numero di immagini maggiore di 1000 per ogni classe.

Per velocizzare il processo di inizializzazione del dataset, è possibile creare all'interno delle directory `train` e `test` un file di cache in formato JSON che prenderà il nome di `cache.json`. La creazione di tale file avviene in automatico quando si chiama il costruttore della classe `NowCastDS` con il parametro `create_cache=True` (*NOTA*: il valore di default è `False`).

```python
training_set = NowCastDS(ds_root_path='/your/ds/root/path', mode='train', create_cache=True)
test_set = NowCastDS(ds_root_path='/your/ds/root/path', mode='test', create_cache=True)
```

<br>

#### 2.2. Struttura dei File `sensor.json`

Considerando una classe con `K` immagini ed `m` valori provenienti dai sensori associati a ciascuna di esse, la struttura del relativo file opzionale `sensors.json` sarà la seguente:

```
{
    "img1_name": [x1_1, x1_2, ..., x1_m],
    "img2_name": [x2_1, x2_2, ..., x2_m],
    ...
    "imgK_name": [xK_1, xK_2, ..., xK_m]
}
```

Si tratta dunque di un file JSON in cui le chiavi sono i percorsi relativi alla directory contenente il file `sensors.json` stesso e i valori sono liste contenente gli `m` dati letti dai sensori per quella specifica immagine.

Si noti che, se si sceglie di inserire il file `sensor.json` all'interno di una directory relativa ad una classe, anche tutte le altre classi devono contenerlo. Qualora per alcune immagini non fossero disponibili uno o più valori relativi ad un sensore, sarà sufficiente inserire al loro posto il valore`nan`. Se ad esempio per l'immagine `img1_name` non si disponesse del valore 2, all'interno del JSON si avrà:

- `"img1_name": [x1_1, null, ..., x1_m]`


<br>

#### 2.3. Verifica della Correttezza della Struttura del Dataset

È possibile verificare la correttezza della struttura del proprio dataset utilizzando lo script `chech_dataset_structure.py` tramite il seguente comando, in cui si indica con `<dataset_path>` il percorso assoluto alla directory principale del dataset:

- `python chech_dataset_structure.py <dataset_path>`

Lo script verificherà la presenza di errori strutturali e li comunicherà all'untento con un apposito messaggio auto-esplicativo. Saranno inoltre forniti avvertimenti  su eventuali aspetti che non si ritengono ottimali per iniziare la procedura di traning; ad esempio potrebbe essere segnalata la presenza di un numero di immagini ritenuto insufficiente per una o più classi.

Gli errori saranno evidenziati con un pallino rosso e la dicitura "ERROR" e andranno necessariamente risolti prima di intraprendere la procedura di allenamento del modello di nowcasting.

Gli avvertimento saranno evidenziati con un pallino giallo e la dicitura "WARNING"; in questo casto non sarà necessario (sebbene caldamente consigliato) risolvere la problematica indicata prima di procedere con l'allenamento del modello.

<br>

#### 2.4. Esempio
In Figura 1.2. si propone un esempio di struttura della directory `train` nel caso di un problema di nowcasting a due classi, in cui, partendo dall'immagine RGB e dai dati di un sensore di temperatura si vuole verificare la presenza o l'assenza di nebbia nell'immagine in ingresso. Le sotto-directory `fog` e `no_fog` contengono rispettivamente immagini con nebbia e immagini in cui la nebbia è assente. Le immagini mostrate in figura sono state acquisite presso l'Osservatorio di Modena.

[--- figura ---]

Un esempio completo che mostra la struttura di un dataset valido, seppur contenente un numero esiguo di immagini, è contenuto all'interno di questo stesso repository:
- `PyNowCast/dataset/example_ds`

*NOTA*: il dataset `example_ds` ha solo uno scopo esemplificativo e non può essere utilizzato per allenare un modello di nowcasting a causa del numero ridotto di immagini presenti.


#### 2.5. Dimensione delle Immagini
È opportuno che tutte le immagini utilizzate per l'allenamento e la verifica del modello di nowcasting abbiano la stessa dimensione. Il package PyNowCast è stato infatti pensato per problemi di nowcasting a camera fissa, quindi si presuppone che tutte le immagini che compongono il training set e il test set provengano dalla stessa camera e presentino di conseguenza le medesime dimensioni.

Qualora questa condizione non sia verificata, è possibile utilizzare lo script `fix_ds_img_shapes` che andrà ad uniformare la dimensioni delle immagini del dataset.

```bash
python fix_ds_img_shapes.py <dataset_root_path> --img_height=<desired_height> --img_width=<desired_width>
```


### 3. Feature Extractor

Il feature extractor è un componente essenziale della maggior parte dei modelli di classificazione basati su reti neurali; il suo compito è, come suggerisce il nome stesso, quello di estrarre una serie di caratteristiche che "riassumano" i tratti salienti dell'oggetto passato in ingresso, che nel nostro caso è un'immagine RGB proveniente da una camera fissa. 

Nell'ambito della Computer Vision, esistono una serie di feature extractor standard che vengono utilizzati per un vasto numero di teask, in quanto risultano molto flessibili e in grado di fornire feature di alto livello che possono venire in contro alle esigenze di problemi anche molto diversi tra loro.

!!!!!!!!!! CITARE !!!!!!!!!!!!!!

<br>

| ![](https://github.com/FabioLanzi/PyNowCast/blob/master/resources/nowcast_fx.jpg) |
| -------------- |
| **Figura 2.** Schema a blocchi dell'autoencoder utilizzato durante la fase di allenamento del feature extractor; il feature extractor vero e proprio è rappresentato dalla sola parte di encoding (ramo "blu" in figura).     |

<br>


Nel nostro caso specifico, tuttavia, risulta più opportuno affidarsi ad un feature extractor più mirato e incentrato in sulle nostre esigenze particolari. Il problema che _PyNowCast_ va ad affrontare infatti è molto vincolato e i vincoli che lo contraddistinguono, se ben sfruttati, possono semplificarlo enormemente. In questo senso, per il nostro package abbiamo deciso di orientarci verso un feature extractor personalizzato basato su un autoencoder, che sia in grado di essere allenato efficacemente in modo non supervisionato.

#### 3.1. Autoencoder

Per adottare questa soluzione ci siamo basati su una semplice osservazione: poiché ci troviamo a trattare immagini provenienti da una camera fissa che riprende una certa porzione di paesaggio, gli elementi che variano tra un'immagine e l'altra sono essenzilamente la condizione metereologica e le condizioni di illuminazione. Fortunatamente ciò che varia sono esattamente le feature che servono ad un classificatore che si occupa di nowcasting. 

In tal senso, l'uso di un autoencoder composto da un encoder e un decoder speculari risulta particolarmente appropriato. Allenando un autoencoder di questo tipo a ricostruire semplicmente le immagini di input all'uscita dell'encoder si avrà un "codice" che rappresneta per l'appunto un riassunto delle immagini di input. Trovando la giusta dimensione di tale codice, l'encoder sarà forzato a rimuovere tutte le informazioni ridondati, che nel nostro caso sono appunto le caratteristiche che non variano tra un'immagine e l'altra; al contempo dovrà preservare gli elementi mutevoli delle medesime (meteo e condizioni di illuminazione).

L'autoencoder utilizzato è mostrato in Figura 2.1.



#### 3.2. Procedura di Allenamento

Per avviare la procedura di allenamento del feature extractor è possibile utilizzare lo script `train_extractor.py` con le seguenti opzioni:

- `--exp_name`: etichetta assegnata alla corrente procedura di allenamento del feature extractor;  per differenziare le etichette assegnate al feature extractor da quelle assegnate al modelo utilizzato per la classificazione, si suggerisce di utilizzare il prefizzo `fx_`. 

- `--ds_root_path`: percorso della directory principale contenete il proprio dataset.
- `--device`: tramite questa opzione è possibile selezionare il device su cui sarà effettuata la procedura di allenamento del feature extractor; i possibili valori sono i seguenti: `'cuda'` o `'cuda:<numero specifica gpu>'` per un allenamento su GPU (consigliato) o `'cpu'` per un allenamento su CPU (sconsigliato). Se non specificato, il valore di default è `'cuda'`

Si propone di seguito un esempio di chiamata:

- `python train_fx.py --exp_name='fx_try1' --ds_root_path='/nas/dataset/nowcast_ds'`



Per gli utenti più esperti, è possibile modificare il file `conf.py` per personalizzare i parametri del training; salvo casi molto particolari, tuttavia, si suggerisce di utilizzare i parametri di default. Per completezza, si riporta la porzione del file di configurazione relativa all'allenamento del feature extractor:


```python
# feature extractor settings
FX_LR = 0.0001  # learning rate used to trane the feature extractor
FX_N_WORKERS = 4  # worker(s) number of the dataloader
FX_BATCH_SIZE = 8  # batch size used to trane the feature extractor
FX_MAX_EPOCHS = 256  # maximum training duration (# epochs)
FX_PATIENCE = 16 # stop training if no improvement is seen for a ‘FX_PATIENCE’ number of epochs
```

### 4. Classificatore

### 5. Risultati