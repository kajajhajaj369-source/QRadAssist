# Dataset selection (Phase 1A)

This document records a **preliminary research decision**. It is not a claim of
experimental results. The dataset has **not** been downloaded in this
repository. Preprocessing, label-mapping, uncertainty handling, and split
protocol are **not** finalized.

## 1. Primary dataset

**CheXpert** is the selected primary dataset for the initial core training and
evaluation task.

CheXpert is a publicly described chest radiograph dataset from Stanford Hospital
(Irvin et al., 2019). The authors report 224,316 chest radiographs from 65,240
patients, collected between October 2002 and July 2017, with labels for 14
observations derived from associated radiology reports by an automated labeler.
Those figures are taken from the dataset publication and official project page;
they have not been independently recounted from a local copy of the data.

This project uses **real-world medical imaging datasets**. AI-generated X-rays
are **not** part of the core dataset.

## 2. Why CheXpert was selected

The following reasons motivated the choice. They are compatibility and
research-design reasons, not performance claims.

- **Multi-label chest X-ray observations.** CheXpert labels multiple findings
  per study rather than a single mutually exclusive class.
- **Coverage of the initial core labels.** Atelectasis, Pneumonia, and Pleural
  Effusion are among CheXpert’s reported observations.
- **Documented uncertainty labels.** The dataset explicitly encodes uncertain
  mentions, which is relevant to a research prototype and must be handled
  explicitly later (policy not chosen yet).
- **Single primary source for the core task.** Using one primary corpus reduces
  the risk of naively mixing hospitals, devices, time periods, and annotation
  pipelines in the training set.
- **Clinical radiographs with reports.** Labels come from a clinical reporting
  process (automated extraction from reports), not from synthetic images.

CheXpert is widely used as a public chest X-ray benchmark. Widespread use is
not, by itself, evidence that a model trained here will be clinically accurate.

## 3. Initial core training labels

The initial **core training task** is multi-label classification of:

1. Pneumonia
2. Atelectasis
3. Pleural Effusion

These correspond to registry ids `pneumonia`, `atelectasis`, and
`pleural_effusion`.

`config/diseases.json` is unchanged in Phase 1A. It still lists `covid19` as a
configured label for later, optional work. Presence in the registry does **not**
mean COVID-19 is part of the core training task or that a model can detect it.

## 4. Why these labels are a multi-label problem

The three findings are **distinct observations**. They are not mutually
exclusive classes of a single diagnosis.

On a chest radiograph, more than one of these observations can be present at
the same time (for example, pleural effusion with atelectasis). CheXpert
assigns each observation separately (positive, negative, uncertain, or not
mentioned, as encoded in the released labels). The learning problem is therefore
**multi-label**: the model should output a score per label, not a single
winner among three categories.

Pneumonia is a clinical diagnosis in practice. CheXpert’s authors included it
as an observation intended to capture report language suggesting primary
infection; that caveat remains relevant when interpreting model scores.

## 5. Why COVID-19 is excluded from the core training task

COVID-19 is **not** added to the core training label set at this stage.

Reasons:

- **Source compatibility.** Public COVID-19 chest X-ray collections typically
  come from different institutions, acquisition systems, time periods, and
  annotation pipelines than CheXpert. Merging them into one training set without
  a controlled protocol can confound “COVID-19” with dataset or source
  identity.
- **CheXpert label set.** COVID-19 is not one of CheXpert’s 14 observations.
  The reported CheXpert collection window ends in July 2017, before COVID-19
  was a clinical entity.
- **Different scientific question.** Detecting COVID-19 from mixed public
  datasets is a different study design than multi-label abnormality
  classification on CheXpert.

The registry may still mention COVID-19 so the software remains extensible.
That is a configuration choice, not a training decision.

## 6. Possible later use of COVID-19

COVID-19 **may** be considered later as an **external validation or
generalization experiment**, only if a scientifically appropriate dataset and
protocol are identified.

Any such experiment would need to:

- keep COVID-19 **out** of the core CheXpert training labels unless a justified
  protocol says otherwise
- state the external data source, labeling method, and population
- address domain shift and source confounding explicitly
- avoid claiming that success on CheXpert abnormalities implies COVID-19
  detection

No COVID-19 dataset has been selected. No protocol has been written.

## 7. Dataset risks

These are known risk classes for CheXpert-style work. They are not measurements
from a local analysis (none has been run).

### Label noise

Training labels are produced by an automated rule-based labeler over free-text
reports, not by exhaustive expert image-by-image annotation of the full
training set. Report language, labeler errors, and the gap between a written
finding and the pixels can all introduce noise.

### Uncertainty labels

CheXpert encodes uncertain mentions (commonly represented as a distinct
uncertain value in the label files). Published work discusses several mapping
policies (for example, treating uncertain as positive, as negative, or ignoring
those examples). **This project has not chosen a policy yet.** Results will
depend on that choice.

### Class imbalance

Published CheXpert documentation reports substantially different positive rates
across observations. Pneumonia is typically much less often labeled positive
than pleural effusion. Imbalance, rare positives, and prevalence shift are
expected evaluation issues. Local class counts have not been computed.

### Patient-level leakage

The authors report many more radiographs than patients, so multiple images and
studies can belong to one person. Splitting by image or study without respecting
patient identity can leak identity, anatomy, and devices into evaluation.
**Patient-level split rules are not finalized.**

### Acquisition and view differences

CheXpert includes multi-view studies (frontal and lateral radiographs are
discussed in the dataset paper). Projection (for example AP vs PA), portable
vs departmental acquisition, laterality, and image quality can affect both
labels and model cues. Which views will be used is **not** decided.

### Domain shift

A model trained on Stanford Hospital radiographs from 2002–2017 may not
transfer to other hospitals, devices, populations, or later time periods.
External COVID-19 data, if used, would be a particularly strong domain-shift
setting. CheXpert’s own expert-labeled validation/test subsets also differ from
the report-derived training labels; that train/test label-generation gap is
itself a form of shift.

## 8. What is not finalized

The following remain open (Phase 1B and later):

- download, storage, and license/compliance workflow
- which CheXpert subset and views to use
- how to map CheXpert columns onto the three core labels
- how to treat blank vs negative vs uncertain labels
- train/validation/test splits, including patient-level constraints
- image validation and preprocessing
- metrics and positive-decision thresholds

No dataset files should be assumed to exist under `data/` yet.

## 9. Data authenticity

The core dataset plan is **real clinical chest radiographs** (CheXpert).
**AI-generated X-rays are not part of the core dataset.**

## References

These are pointers to the dataset’s originating documentation, not a
literature review and not results of this project.

- Irvin, J., et al. *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.* AAAI, 2019. https://arxiv.org/abs/1901.07031
- CheXpert project page: https://stanfordmlgroup.github.io/competitions/chexpert/
