-- Pandoc filter: typeset every table as a floating three-line (booktabs) table.
-- A float never breaks across pages. Wide tables span the text width and wrap their long text columns;
-- narrow tables keep their natural width. Short columns are centred; text columns are left-aligned.

local CHAR_CM = 0.17          -- approximate width of one character in \small Times, in cm

local function cell_text(cell)
  return pandoc.utils.stringify(cell.contents)
end

local function cell_tex(cell)
  local s = pandoc.write(pandoc.Pandoc(cell.contents), 'latex')
  s = s:gsub('%s+$', ''):gsub('\n\n+', ' \\newline '):gsub('\n', ' ')
  return s
end

local function ulen(s)
  return utf8.len(s) or #s
end

local function row_tex(row)
  local cells = {}
  for _, c in ipairs(row.cells) do table.insert(cells, cell_tex(c)) end
  return table.concat(cells, ' & ') .. ' \\\\'
end

function Table(tbl)
  local head, body = {}, {}
  for _, r in ipairs(tbl.head.rows) do table.insert(head, r) end
  for _, b in ipairs(tbl.bodies) do
    for _, r in ipairs(b.head) do table.insert(body, r) end
    for _, r in ipairs(b.body) do table.insert(body, r) end
  end
  for _, r in ipairs(tbl.foot.rows) do table.insert(body, r) end

  local ncol = #tbl.colspecs
  local hlen, blen = {}, {}
  for j = 1, ncol do hlen[j], blen[j] = 0, 0 end
  for _, r in ipairs(head) do
    for j, c in ipairs(r.cells) do hlen[j] = math.max(hlen[j], ulen(cell_text(c))) end
  end
  for _, r in ipairs(body) do
    for j, c in ipairs(r.cells) do blen[j] = math.max(blen[j], ulen(cell_text(c))) end
  end

  -- width each column needs: its longest body cell, or half its header (headers may wrap onto two lines)
  local need, total = {}, 0
  for j = 1, ncol do
    need[j] = math.max(blen[j], math.ceil(hlen[j] / 2))
    total = total + need[j]
  end
  local wide = total > 80

  local spec, xcols, xsum = {}, {}, 0
  -- long text columns share the width; weights grow with the square root of their longest cell
  local weight = {}
  for j = 1, ncol do
    if wide and blen[j] > 24 then
      xcols[j] = true
      weight[j] = math.sqrt(blen[j])
      xsum = xsum + weight[j]
    end
  end
  local nx = 0
  for _ in pairs(xcols) do nx = nx + 1 end
  for j = 1, ncol do
    if xcols[j] then
      spec[j] = string.format('>{\\RaggedRight\\arraybackslash\\hsize=%.2f\\hsize}X', weight[j] / xsum * nx)
    elseif hlen[j] > blen[j] + 4 and hlen[j] > 14 then
      -- short values under a long header: a fixed-width column whose header wraps
      local cm = math.max(blen[j], math.ceil(hlen[j] / 2)) * CHAR_CM + 0.3
      local align = (j == 1) and '\\RaggedRight' or '\\centering'
      spec[j] = string.format('>{%s\\arraybackslash}p{%.2fcm}', align, cm)
    elseif j == 1 or blen[j] > 18 then
      spec[j] = 'l'
    else
      spec[j] = 'c'
    end
  end

  local env = (nx > 0) and 'tabularx' or 'tabular'
  local width = (nx > 0) and '{\\linewidth}' or ''
  local caption = ''
  if #tbl.caption.long > 0 then
    caption = pandoc.write(pandoc.Pandoc(tbl.caption.long), 'latex'):gsub('%s+$', ''):gsub('\n', ' ')
  end

  local out = {}
  table.insert(out, '\\begin{table}[htbp]')
  table.insert(out, '\\centering\\small')
  if caption ~= '' then table.insert(out, '\\caption{' .. caption .. '}') end
  table.insert(out, string.format('\\begin{%s}%s{%s}', env, width, table.concat(spec, '')))
  table.insert(out, '\\toprule')
  for _, r in ipairs(head) do table.insert(out, row_tex(r)) end
  table.insert(out, '\\midrule')
  for _, r in ipairs(body) do table.insert(out, row_tex(r)) end
  table.insert(out, '\\bottomrule')
  table.insert(out, string.format('\\end{%s}', env))
  table.insert(out, '\\end{table}')
  return pandoc.RawBlock('latex', table.concat(out, '\n'))
end
