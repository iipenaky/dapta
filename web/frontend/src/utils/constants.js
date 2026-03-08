/**
 * @fileoverview App-wide constants. Single source of truth — never duplicate.
 */

export const METRIC_LIST = [
  { key: 'ciu_rate',             label: 'Informativeness (CIU)', color: 'var(--accent-blue)'   },
  { key: 'mc_score',             label: 'Main Concepts (MC)',     color: 'var(--accent-teal)'   },
  { key: 'mlu_morphemes',        label: 'Sentence Length (MLU)', color: 'var(--accent-green)'  },
  { key: 'ttr',                  label: 'Vocabulary (TTR)',       color: 'var(--accent-amber)'  },
  { key: 'syntactic_complexity', label: 'Grammar Complexity',    color: 'var(--accent-purple)' },
]

export const SESSION_STEPS = [
  { id: 'assess',    label: 'Assess'    },
  { id: 'recommend', label: 'Recommend' },
  { id: 'exercise',  label: 'Exercise'  },
  { id: 'feedback',  label: 'Feedback'  },
]

export const ACCEPTED_FILES = '.cha,.wav,.mp3,.mp4,.m4a,.webm,.ogg'

export const OVERALL_RATING_COLOURS = {
  excellent:      'var(--accent-green)',
  good:           'var(--accent-teal)',
  stable:         'var(--accent-amber)',
  needs_practice: 'var(--accent-blue)',
}

export const CONFIDENCE_COLOURS = {
  High:     'var(--accent-green)',
  Moderate: 'var(--accent-amber)',
  Low:      'var(--accent-red)',
}

export const APHASIA_SUBTYPES = [
  'Broca', 'Wernicke', 'Anomic', 'Conduction', 'Global', 'Other',
]
