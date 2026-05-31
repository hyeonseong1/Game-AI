/** Difficulty presets — simulates RL agent reaction speed & accuracy */
const AI_DIFFICULTY = {
  easy: {
    label: "Easy",
    reactionDelay: 280,
    mistakeRate: 0.35,
    speedMult: 0.55,
    accuracy: 0.5,
  },
  medium: {
    label: "Medium",
    reactionDelay: 140,
    mistakeRate: 0.15,
    speedMult: 0.8,
    accuracy: 0.72,
  },
  hell: {
    label: "Hell",
    reactionDelay: 40,
    mistakeRate: 0.03,
    speedMult: 1.15,
    accuracy: 0.95,
  },
};

function getDifficulty(key) {
  return AI_DIFFICULTY[key] || AI_DIFFICULTY.medium;
}

function calcAiPercent(playerScore, aiScore) {
  if (aiScore <= 0 && playerScore <= 0) return 50;
  if (aiScore <= 0) return 100;
  const pct = Math.round((playerScore / aiScore) * 100);
  return Math.min(150, Math.max(0, pct));
}

function shouldAiMistake(difficulty) {
  return Math.random() < getDifficulty(difficulty).mistakeRate;
}
