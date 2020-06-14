
## Organizzazione del Dataset

Come prima cosa è necessario organizzare i dati sui quali si vuole allenare il modello di nowcasting secondo il semplice schema mostrato in Figura 1.1. 
Si avrà pertanto una directory principale, indicata con `<dataset_name>`, con un nome a piacere, che dovrà necessariamente contenere due sotto-directory denominate rispettivamente `train` e `test`, che a loro volta devono contenere le varie sotto-directory contenenti le immagini suddivise per classi. 

<br>

| ![](https://github.com/FabioLanzi/PyNowCast/blob/master/resources/ds_tree.jpg) |
| ------------------------------------------------------------ |
| **Figura 1.1.** Struttura della directory contenente il dataset. I nomi indicati in verde in figura possono essere scelti a piacere. |

<br>



### Directory di Train e Test

La directory `train`, come il nome suggerisce, è quella preposta a ospitare le immagini di training, ovvero quelle sulle quali sarà allenato il modello di nowcasting; allo stesso modo, la directory `test` conterrà le immagini utilizzate per valutare le prestazioni del suddetto modello. Si noti che, per ottenere una valutazione corretta delle prestazioni del modello, gli insiemi composti dalle immagini di training e dalle immagini di test dovrebbero essere completamente disgiunti, quindi privi di immagini comuni.

Le sotto-directory `train` e `test` devono contenere a loro volta `n` sotto-directory, dove `n` (con `n` maggiore o uguale a 2) è il numero di classi scelte per lo specifico problema di nowcasting che si vuole approcciate con il presente framework. La directory di ogni classe deve contenere esclusivamente le immagini relative a quella specifica classe, affiancate al più da un file `sensors.json` contenente le informazioni relative ad eventuali dati aggiuntivi provenienti da appositi sensori (esempio: sensori di umidità, temperatura, pressione, ...).

Per allenare correttamente il modello di nowcasting, si consiglia di avere un training set bilanciato, quindi con un quantitativo di immagini simile per ogni classe; si consiglia inoltre di avere un numero di immagini maggiore di 1000 per ogni classe.


<br>

### Struttura dei File `sensor.json`

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

### Verifica della Correttezza della Struttura del Dataset

È possibile verificare la correttezza della struttura del proprio dataset utilizzando lo script `chech_dataset_structure.py` tramite il seguente comando, in cui si indica con `<dataset_path>` il percorso assoluto alla directory principale del dataset:

- `python chech_dataset_structure.py <dataset_path>`

Lo script verificherà la presenza di errori strutturali e li comunicherà all'untento con un apposito messaggio auto-esplicativo. Saranno inoltre forniti avvertimenti  su eventuali aspetti che non si ritengono ottimali per iniziare la procedura di traning; ad esempio potrebbe essere segnalata la presenza di un numero di immagini ritenuto insufficiente per una o più classi.

Gli errori saranno evidenziati con un pallino rosso e la dicitura "ERROR" e andranno necessariamente risolti prima di intraprendere la procedura di allenamento del modello di nowcasting.

Gli avvertimento saranno evidenziati con un pallino giallo e la dicitura "WARNING"; in questo casto non sarà necessario (sebbene caldamente consigliato) risolvere la problematica indicata prima di procedere con l'allenamento del modello.

<br>

### Esempio Directory `train`
In Figura 1.2. si propone un esempio di struttura della directory `train` nel caso di un problema di nowcasting a due classi, in cui, partendo dall'immagine RGB e dai dati di un sensore di temperatura si vuole verificare la presenza o l'assenza di nebbia nell'immagine in ingresso. Le sotto-directory `fog` e `no_fog` contengono rispettivamente immagini con nebbia e immagini in cui la nebbia è assente. Le immagini mostrate in figura sono state acquisite presso l'Osservatorio di Modena.

[--- figura ---]
