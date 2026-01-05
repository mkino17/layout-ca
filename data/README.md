## Data

To process your own document pairs (PDF format), create language-specific folders using language codes and place the corresponding files inside. By default, folders for `en` and `ja` are already prepared. See [../codes/ocr_object_detection.py](codes/ocr_object_detection.py) for the language codes.

For evaluation and development, we provide a curated dataset derived from bilingual reports published by UNESCO. This corpus serves as a reference dataset for testing and benchmarking the pipeline. Each dataset is stored in `data/to_eval` and `data/to_dev`.

&nbsp;

<p align="center">
  <img src="../fig/sample-en.jpg" width="25%">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="../fig/sample-en.jpg" width="25%">
</p>
<p align="center">
  <em>Figure 3: Source (left) and target (right) document images.</em>
</p>



Link to the PDF

Evaluation Dataset: [English](data/to_eval/pdf/unesco/en/unesco_en_eval.pdf),[Japanese](data/to_eval/pdf/unesco/ja/unesco_ja_eval.pdf), [Japanese(shuffled)](data/to_eval/pdf/shuffled_unesco/ja/shuffled_unesco_ja_eval.pdf)

Development Dataset: [English](data/to_dev/pdf/unesco_en_dev.pdf), [Japanese](data/to_dev/pdf/unesco_ja_dev.pdf)



Original version © UNESCO.

The present work is not an official UNESCO publication and shall not be considered as such.  Masaki Kinouchi edited the publications with permission from UNESCO. Added texts are enclosed in a gray box, and original photographs, illustrations, logos and figures have been covered to preserve copyright. This material is available in Open Access under the Attribution ShareAlike 3.0 IGO (CC-BY-SA 3.0 IGO) license (http://creativecommons.org/licenses/by-sa/3.0/igo/).