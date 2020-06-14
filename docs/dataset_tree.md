
## Organizzazione del Dataset

Come prima cosa è necessario organizzare i dati sui quali si vuole allenare il modello di nowcasting secondo il semplice schema mostrato in Figura 1.1. 
Si avrà pertanto una directory principale, indicata con `<dataset_name>`, con un nome a piacere, che dovrà necessariamente contenere due sotto-directory denominate rispettivamente `train` e `test`, che a loro volta devono contenere le varie sotto-directory contenenti le immagini suddivise per classi. 

<br>

| ![](https://github.com/FabioLanzi/PyNowCast/blob/master/resources/ds_tree.jpg) |
| ------------------------------------------------------------ |
| **Figura 1.1.** Struttura della directory contenente il dataset. I nomi indicati in verde in figura possono essere scelti a piacere. |

<br>

La directory `train`, come il nome suggerisce, è quella preposta a ospitare le immagini di training, ovvero quelle sulle quali sarà allenato il modello di nowcasting; allo stesso modo, la directory `test` conterrà le immagini utilizzate per valutare le prestazioni del suddetto modello. Si noti che, per ottenere una valutazione corretta delle prestazioni del modello, gli insiemi composti dalle immagini di training e dalle immagini di test dovrebbero essere completamente disgiunti, quindi privi di immagini comuni.

...

È possibile verificare la correttezza della struttura del proprio dataset utilizzando lo script `chech_dataset_structure.py` tramite il seguente comando, in cui si indica con `<dataset_path>` il percorso assoluto alla directory principale del dataset:

- `python chech_dataset_structure.py <dataset_path>`

