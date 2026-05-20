export type FlagNoteTypo = {
  word: string;
  suggestion: string;
};

/** Common misspellings → correction (checked before fuzzy match). */
const COMMON_MISSPELLINGS: Record<string, string> = {
  teh: 'the',
  adn: 'and',
  taht: 'that',
  wiht: 'with',
  fro: 'for',
  fo: 'of',
  ot: 'to',
  form: 'from',
  recieve: 'receive',
  occured: 'occurred',
  seperate: 'separate',
  definately: 'definitely',
  appex: 'apex',
  apix: 'apex',
  noduel: 'nodule',
  noduels: 'nodules',
  noulde: 'nodule',
  tuberclosis: 'tuberculosis',
  tuberculosois: 'tuberculosis',
  pnuemonia: 'pneumonia',
  pneunomia: 'pneumonia',
  consolodation: 'consolidation',
  consoliation: 'consolidation',
  pleural: 'pleural',
  effusion: 'effusion',
  effusuon: 'effusion',
  opacitiy: 'opacity',
  opacities: 'opacities',
  infiltrate: 'infiltrate',
  infilitrate: 'infiltrate',
  cavitation: 'cavitation',
  cavitations: 'cavitations',
  mediastinal: 'mediastinal',
  mediastinum: 'mediastinum',
  hilar: 'hilar',
  hilum: 'hilum',
  apical: 'apical',
  lateral: 'lateral',
  bilateral: 'bilateral',
  unilateral: 'unilateral',
  posterior: 'posterior',
  anterior: 'anterior',
  superior: 'superior',
  inferior: 'inferior',
  peripheral: 'peripheral',
  central: 'central',
  region: 'region',
  regions: 'regions',
  overlay: 'overlay',
  heatmap: 'heatmap',
  saliency: 'saliency',
  artifact: 'artifact',
  artifacts: 'artifacts',
  motion: 'motion',
  underexposed: 'underexposed',
  overexposed: 'overexposed',
  rotated: 'rotated',
  cropped: 'cropped',
  obscured: 'obscured',
};

/** Words treated as correctly spelled in anomaly notes (radiology + common English). */
const ALLOWLIST = new Set(
  [
    ...Object.values(COMMON_MISSPELLINGS),
    'a',
    'an',
    'and',
    'are',
    'as',
    'at',
    'be',
    'by',
    'for',
    'from',
    'has',
    'have',
    'in',
    'is',
    'it',
    'its',
    'may',
    'no',
    'not',
    'of',
    'on',
    'or',
    'the',
    'this',
    'to',
    'was',
    'with',
    'without',
    'vs',
    'versus',
    'likely',
    'possible',
    'probably',
    'suspected',
    'suspect',
    'missed',
    'missing',
    'false',
    'true',
    'positive',
    'negative',
    'tb',
    'tuberculosis',
    'chest',
    'xray',
    'ray',
    'lung',
    'lungs',
    'lobe',
    'lobes',
    'upper',
    'lower',
    'left',
    'right',
    'mid',
    'middle',
    'zone',
    'field',
    'apex',
    'apices',
    'base',
    'bases',
    'hilum',
    'hilar',
    'mediastinal',
    'mediastinum',
    'cardiac',
    'heart',
    'pleural',
    'effusion',
    'consolidation',
    'opacity',
    'opacities',
    'infiltrate',
    'infiltration',
    'infiltrates',
    'cavity',
    'cavitation',
    'cavitations',
    'nodule',
    'nodules',
    'mass',
    'masses',
    'lesion',
    'lesions',
    'finding',
    'findings',
    'region',
    'regions',
    'area',
    'areas',
    'segment',
    'segmentation',
    'mask',
    'masked',
    'heatmap',
    'overlay',
    'saliency',
    'cam',
    'score',
    'risk',
    'high',
    'low',
    'quality',
    'image',
    'images',
    'study',
    'view',
    'pa',
    'lateral',
    'ap',
    'cxr',
    'film',
    'clipping',
    'artifact',
    'artifacts',
    'noise',
    'blur',
    'rotation',
    'cropped',
    'edge',
    'edges',
    'costophrenic',
    'diaphragm',
    'rib',
    'ribs',
    'spine',
    'scapula',
    'clavicle',
    'soft',
    'tissue',
    'bone',
    'bony',
    'calcification',
    'fibrosis',
    'scarring',
    'pneumothorax',
    'atelectasis',
    'pneumonia',
    'miliary',
    'cavitary',
    'apical',
    'basal',
    'posterior',
    'anterior',
    'superior',
    'inferior',
    'peripheral',
    'central',
    'bilateral',
    'unilateral',
    'subtle',
    'prominent',
    'marked',
    'mild',
    'moderate',
    'severe',
    'small',
    'large',
    'tiny',
    'dominant',
    'suspicious',
    'consistent',
    'inconsistent',
    'wrong',
    'incorrect',
    'correct',
    'expected',
    'unexpected',
    'model',
    'prediction',
    'predicted',
  ].map((w) => w.toLowerCase()),
);

function levenshtein(a: string, b: string): number {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;
  const row = new Array<number>(b.length + 1);
  for (let j = 0; j <= b.length; j += 1) row[j] = j;
  for (let i = 1; i <= a.length; i += 1) {
    let prev = row[0];
    row[0] = i;
    for (let j = 1; j <= b.length; j += 1) {
      const temp = row[j];
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      row[j] = Math.min(row[j] + 1, row[j - 1] + 1, prev + cost);
      prev = temp;
    }
  }
  return row[b.length];
}

function closestAllowlistWord(word: string): { suggestion: string; distance: number } | null {
  let best: { suggestion: string; distance: number } | null = null;
  const maxDist = word.length <= 5 ? 1 : 2;
  for (const candidate of ALLOWLIST) {
    if (Math.abs(candidate.length - word.length) > maxDist) continue;
    const distance = levenshtein(word, candidate);
    if (distance === 0 || distance > maxDist) continue;
    if (!best || distance < best.distance) {
      best = { suggestion: candidate, distance };
    }
  }
  return best;
}

/**
 * Return possible typos in a free-text anomaly note.
 * Empty notes return no issues.
 */
export function findFlagNoteTypos(note: string): FlagNoteTypo[] {
  const trimmed = note.trim();
  if (!trimmed) return [];

  const seen = new Set<string>();
  const issues: FlagNoteTypo[] = [];
  const tokens = trimmed.match(/\b[A-Za-z][A-Za-z'-]*\b/g) ?? [];

  for (const raw of tokens) {
    const lower = raw.toLowerCase();
    if (lower.length < 3) continue;
    if (ALLOWLIST.has(lower)) continue;
    if (seen.has(lower)) continue;

    const mapped = COMMON_MISSPELLINGS[lower];
    if (mapped) {
      seen.add(lower);
      issues.push({ word: raw, suggestion: mapped });
      continue;
    }

    const close = closestAllowlistWord(lower);
    if (close) {
      seen.add(lower);
      issues.push({ word: raw, suggestion: close.suggestion });
    }
  }

  return issues;
}
