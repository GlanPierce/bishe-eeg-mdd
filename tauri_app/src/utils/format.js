export function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '--';
  return Number(value).toFixed(digits);
}

export function signedFmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '--';
  const number = Number(value);
  return `${number >= 0 ? '+' : ''}${number.toFixed(digits)}`;
}
