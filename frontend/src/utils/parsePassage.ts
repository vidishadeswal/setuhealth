/**
 * Parses a citation's passage content for display. Chunk content is stored with all
 * whitespace (including original newlines) collapsed to single spaces (see
 * backend/app/ingestion/chunker.py's word-join), so a markdown table embedded in a
 * source document survives only as pipe-delimited text with no row breaks.
 *
 * Recovery relies on one property that always holds for the tables this corpus
 * generates: every cell is non-empty (missing values are written as "NA", never
 * blank), and adjacent pipes at a row boundary have nothing but whitespace between
 * them. Splitting the whole pipe-delimited region on "|" therefore yields an empty
 * string exactly at each row boundary — a "---"-only run marks the header separator
 * row — so the flat cell list can be re-grouped into rows without needing the
 * original newlines at all.
 */

export interface ParsedTable {
  headers: string[];
  rows: string[][];
}

export interface ParsedPassage {
  prefix: string;
  table: ParsedTable | null;
  suffix: string;
}

export function parsePassageContent(content: string): ParsedPassage {
  const firstPipe = content.indexOf('|');
  const lastPipe = content.lastIndexOf('|');

  if (firstPipe === -1 || lastPipe === firstPipe) {
    return { prefix: content, table: null, suffix: '' };
  }

  const prefix = content.slice(0, firstPipe).trim();
  const suffix = content.slice(lastPipe + 1).trim();
  const tableRegion = content.slice(firstPipe, lastPipe + 1);

  const cells = tableRegion
    .split('|')
    .map((c) => c.trim())
    .filter((_, i, arr) => !(i === 0 || i === arr.length - 1)); // drop outer empty edges

  const rows: string[][] = [[]];
  for (const cell of cells) {
    if (cell === '') {
      rows.push([]);
    } else {
      rows[rows.length - 1].push(cell);
    }
  }
  const nonEmptyRows = rows.filter((r) => r.length > 0);

  if (nonEmptyRows.length < 2) {
    // Not enough structure to call this a real table — fall back to plain text.
    return { prefix: content, table: null, suffix: '' };
  }

  const [headers, ...rest] = nonEmptyRows;
  const bodyRows = rest.filter((r) => !r.every((c) => /^-+$/.test(c)));
  const columnCount = headers.length;

  return {
    prefix,
    table: { headers, rows: bodyRows.filter((r) => r.length === columnCount) },
    suffix,
  };
}
