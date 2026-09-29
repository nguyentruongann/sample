/* The dashboard only reads PostgreSQL through its own read-only API. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const svgNS = 'http://www.w3.org/2000/svg';
  const number = new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 1 });
  const fullTime = new Intl.DateTimeFormat('vi-VN', {
    timeZone: 'Asia/Ho_Chi_Minh', day: '2-digit', month: '2-digit',
    hour: '2-digit', minute: '2-digit', hour12: false,
  });
  const shortTime = new Intl.DateTimeFormat('vi-VN', {
    timeZone: 'Asia/Ho_Chi_Minh', hour: '2-digit', minute: '2-digit', hour12: false,
  });
  const clockFormat = new Intl.DateTimeFormat('vi-VN', {
    timeZone: 'Asia/Ho_Chi_Minh', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
  });
  const state = {
    mode: 'Car', horizon: 10, layer: 'forecast', time: 'live', selected: null,
    snapshot: null, request: 0, detailRequest: 0, seconds: 30,
    zoom: 11, centerLon: 106.7009, centerLat: 10.7769, fitted: false,
    polygons: new Map(), tiles: new Map(),
  };
  const tileSize = 256;
  const minZoom = 9;
  const maxZoom = 16;

  const svgEl = (tag, attrs = {}) => {
    const node = document.createElementNS(svgNS, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
    return node;
  };
  const time = (value) => value ? fullTime.format(new Date(value)) : '—';
  const short = (value) => value ? shortTime.format(new Date(value)) : '—';
  const addMinutes = (value, minutes) => new Date(new Date(value).getTime() + minutes * 60000).toISOString();
  const fmt = (value) => number.format(value);

  function status(kind, label, detail) {
    $('statusDot').className = 'status-dot ' + kind;
    $('statusLabel').textContent = label;
    $('statusDetail').textContent = detail;
  }

  function updateFreshness() {
    const snapshot = state.snapshot;
    if (!snapshot) return;
    if (state.time !== 'live') {
      status('stale', 'ĐANG XEM LỊCH SỬ', 'Chọn Mới nhất để theo dõi trực tiếp');
      return;
    }
    const ageMinutes = (Date.now() - new Date(snapshot.generated_at_utc).getTime()) / 60000;
    if (ageMinutes > 25) status('stale', 'DỰ BÁO ĐÃ CŨ', `Lượt gần nhất: ${time(snapshot.generated_at_utc)}`);
    else status('live', 'ĐANG THEO DÕI', `Cập nhật lúc ${short(snapshot.generated_at_utc)} · ICT`);
  }

  function empty(message) {
    state.snapshot = null;
    state.polygons.clear();
    state.fitted = false;
    $('hexLayer').replaceChildren();
    $('selectedLayer').replaceChildren();
    $('mapEmpty').hidden = false;
    $('mapEmpty').querySelector('span').textContent = message;
    $('forecastTime').textContent = $('forecastEnd').textContent = $('observedTime').textContent = '—';
    $('modelVersion').textContent = '—';
    for (const id of ['kpiTotal', 'kpiPeak', 'kpiCells', 'kpiObserved', 'chartValue']) $(id).textContent = '—';
    $('hotspots').replaceChildren();
    $('trendChart').replaceChildren();
    $('horizonDetail').replaceChildren();
    $('legendP95').textContent = '—';
  }

  async function load() {
    const request = ++state.request;
    const params = new URLSearchParams({ travel_mode: state.mode, horizon_minutes: state.horizon });
    if (state.time !== 'live') params.set('forecast_start_utc', state.time);
    status('', 'ĐANG ĐỒNG BỘ', 'Đang đọc kết quả mới nhất');
    try {
      const response = await fetch('/v1/dashboard/snapshot?' + params, { cache: 'no-store' });
      if (!response.ok) {
        if (response.status === 404) throw new Error('Chưa có dự báo. Hãy đợi luồng dự báo chạy xong.');
        throw new Error('Không thể lấy dữ liệu từ API (' + response.status + ').');
      }
      const snapshot = await response.json();
      if (request !== state.request) return;
      state.snapshot = snapshot;
      $('mapEmpty').hidden = !!snapshot.cells.length;
      if (!snapshot.cells.length) $('mapEmpty').querySelector('span').textContent = 'Các mã H3 trong batch không hợp lệ hoặc chưa có hình học.';
      render(snapshot);
      updateFreshness();
    } catch (error) {
      if (request !== state.request) return;
      empty(error.message);
      status('error', 'CHƯA CÓ DỮ LIỆU', error.message);
    }
  }

  function setActive(container, dataKey, value) {
    for (const button of $(container).querySelectorAll('button')) {
      button.classList.toggle('active', button.dataset[dataKey] === String(value));
    }
  }

  function renderTimes(snapshot) {
    const select = $('timeSelect');
    select.replaceChildren(new Option('● Mới nhất', 'live'));
    const times = [...snapshot.available_times];
    if (state.time !== 'live' && !times.includes(state.time)) times.push(state.time);
    for (const value of times) select.add(new Option(time(value) + ' · ICT', value));
    select.value = state.time;
  }

  function render(snapshot) {
    renderTimes(snapshot);
    $('forecastTime').textContent = time(snapshot.forecast_start_utc);
    $('forecastEnd').textContent = time(addMinutes(snapshot.forecast_start_utc, state.horizon));
    $('observedTime').textContent = time(snapshot.observed_at_utc);
    $('modelVersion').textContent = snapshot.model_version.slice(0, 8);
    const cells = snapshot.cells;
    const peak = cells.reduce((best, item) => Math.max(best, item.predicted_demand), 0);
    const total = cells.reduce((sum, item) => sum + item.predicted_demand, 0);
    $('kpiTotal').textContent = fmt(total);
    $('kpiTotalSub').textContent = `Tổng ${state.horizon} phút tới · ${state.mode}`;
    $('kpiPeak').textContent = fmt(peak);
    $('kpiPeakSub').textContent = `Trong một ô · ${state.horizon} phút tới`;
    $('kpiCells').textContent = fmt(cells.length);
    $('kpiObserved').textContent = snapshot.observed_at_utc ? fmt(snapshot.observed_total) : '—';
    renderMap();
    renderHotspots();
    const selected = cells.find((item) => item.hex_id_7 === state.selected);
    if (selected) selectCell(state.selected);
    else {
      state.selected = null;
      resetDetail();
      renderChart(snapshot.region_history, 'Toàn vùng');
    }
  }

  function percentile(values, p) {
    if (!values.length) return 1;
    const sorted = [...values].sort((a, b) => a - b);
    return Math.max(sorted[Math.floor((sorted.length - 1) * p)], 1);
  }

  const colors = ['#25485d', '#1a7d91', '#38d7c7', '#b7e17b', '#ffb65e', '#ee665f'];
  function color(value, scale) {
    if (value === null || value === undefined) return '#284052';
    const fraction = Math.min(1, Math.sqrt(Math.max(0, value) / scale));
    const slot = Math.min(colors.length - 2, Math.floor(fraction * (colors.length - 1)));
    const amount = fraction * (colors.length - 1) - slot;
    const first = colors[slot].slice(1).match(/../g).map((x) => parseInt(x, 16));
    const next = colors[slot + 1].slice(1).match(/../g).map((x) => parseInt(x, 16));
    return '#' + first.map((part, i) => Math.round(part + (next[i] - part) * amount).toString(16).padStart(2, '0')).join('');
  }

  // The tiles and H3 polygons use the same Web Mercator projection. API
  // boundaries are [longitude, latitude]; the map stays aligned while panning.
  function world([lon, lat], zoom = state.zoom) {
    const size = tileSize * 2 ** zoom;
    const sine = Math.sin(Math.max(-85.0511, Math.min(85.0511, lat)) * Math.PI / 180);
    return [(lon + 180) / 360 * size,
      (0.5 - Math.log((1 + sine) / (1 - sine)) / (4 * Math.PI)) * size];
  }

  function geographic([x, y], zoom = state.zoom) {
    const size = tileSize * 2 ** zoom;
    return [x / size * 360 - 180,
      Math.atan(Math.sinh(Math.PI * (1 - 2 * y / size))) * 180 / Math.PI];
  }

  function viewport() {
    const stage = $('mapStage');
    return [stage.clientWidth, stage.clientHeight];
  }

  function fitMap(cells) {
    if (!cells.length) return;
    const points = cells.flatMap((cell) => cell.boundary.map((coordinate) => world(coordinate, 0)));
    const xs = points.map((point) => point[0]);
    const ys = points.map((point) => point[1]);
    const [width, height] = viewport();
    const boundsWidth = Math.max(...xs) - Math.min(...xs);
    const boundsHeight = Math.max(...ys) - Math.min(...ys);
    const scale = Math.min(width * .88 / Math.max(boundsWidth, .00001),
      height * .84 / Math.max(boundsHeight, .00001));
    state.zoom = Math.max(minZoom, Math.min(maxZoom, Math.floor(Math.log2(scale))));
    [state.centerLon, state.centerLat] = geographic([
      (Math.min(...xs) + Math.max(...xs)) / 2,
      (Math.min(...ys) + Math.max(...ys)) / 2,
    ], 0);
    state.fitted = true;
  }

  function renderTiles() {
    const [width, height] = viewport();
    const [centerX, centerY] = world([state.centerLon, state.centerLat]);
    const count = 2 ** state.zoom;
    const wanted = new Set();
    const layer = $('baseMapTiles');
    const firstX = Math.floor((centerX - width / 2) / tileSize);
    const lastX = Math.floor((centerX + width / 2) / tileSize);
    const firstY = Math.max(0, Math.floor((centerY - height / 2) / tileSize));
    const lastY = Math.min(count - 1, Math.floor((centerY + height / 2) / tileSize));
    for (let x = firstX; x <= lastX; x++) {
      for (let y = firstY; y <= lastY; y++) {
        const wrappedX = ((x % count) + count) % count;
        const key = `${state.zoom}/${wrappedX}/${y}`;
        wanted.add(key);
        let tile = state.tiles.get(key);
        if (!tile) {
          tile = document.createElement('img');
          tile.alt = '';
          tile.draggable = false;
          tile.src = `https://tile.openstreetmap.org/${key}.png`;
          layer.append(tile);
          state.tiles.set(key, tile);
        }
        tile.style.left = `${x * tileSize - centerX + width / 2}px`;
        tile.style.top = `${y * tileSize - centerY + height / 2}px`;
      }
    }
    for (const [key, tile] of state.tiles) {
      if (!wanted.has(key)) { tile.remove(); state.tiles.delete(key); }
    }
  }

  function resetMap() {
    state.fitted = false;
    if (state.snapshot?.cells.length) renderMap();
  }

  function renderMap() {
    const snapshot = state.snapshot;
    const cells = snapshot.cells;
    const layer = $('hexLayer');
    layer.replaceChildren();
    $('selectedLayer').replaceChildren();
    state.polygons.clear();
    if (!cells.length) return;
    if (!state.fitted) fitMap(cells);
    const [width, height] = viewport();
    $('hexMap').setAttribute('viewBox', `0 0 ${width} ${height}`);
    renderTiles();
    const [centerX, centerY] = world([state.centerLon, state.centerLat]);
    const project = (coordinate) => {
      const [x, y] = world(coordinate);
      return [x - centerX + width / 2, y - centerY + height / 2];
    };
    const value = (cell) => state.layer === 'forecast' ? cell.predicted_demand : cell.observed_demand;
    const p95 = percentile(cells.map(value).filter((x) => x !== null), .95);
    const ordered = [...cells].sort((a, b) => (value(a) ?? -1) - (value(b) ?? -1));
    for (const cell of ordered) {
      const points = cell.boundary.map((point) => project(point).map((x) => x.toFixed(2)).join(',')).join(' ');
      const polygon = svgEl('polygon', { points, fill: color(value(cell), p95), class: 'hex-cell' });
      polygon.dataset.hex = cell.hex_id_7;
      polygon.addEventListener('pointermove', (event) => showTooltip(event, cell));
      polygon.addEventListener('pointerleave', () => { $('mapTooltip').hidden = true; });
      layer.append(polygon);
      state.polygons.set(cell.hex_id_7, polygon);
    }
    $('legendP95').textContent = 'P95 ≈ ' + fmt(p95) + ' chuyến';
    $('mapNote').textContent = state.layer === 'forecast'
      ? `Dự báo tổng nhu cầu từ ${short(snapshot.forecast_start_utc)} đến ${short(addMinutes(snapshot.forecast_start_utc, state.horizon))} · ${state.horizon} phút.`
      : `Nhu cầu đã quan sát tại bucket ${time(snapshot.observed_at_utc)} · một bucket 10 phút.`;
    if (state.selected) highlightCell(state.selected);
  }

  function showTooltip(event, cell) {
    const tooltip = $('mapTooltip');
    const rect = $('mapStage').getBoundingClientRect();
    const current = state.layer === 'forecast' ? cell.predicted_demand : cell.observed_demand;
    tooltip.replaceChildren();
    const title = document.createElement('strong');
    title.textContent = cell.hex_id_7;
    const line = document.createElement('span');
    line.textContent = current == null ? 'Không có bucket quan sát' : fmt(current) + ' chuyến · ' + (state.layer === 'forecast' ? 'dự báo' : 'đã quan sát');
    tooltip.append(title, line);
    tooltip.style.left = Math.min(event.clientX - rect.left, rect.width - 195) + 'px';
    tooltip.style.top = Math.max(55, event.clientY - rect.top) + 'px';
    tooltip.hidden = false;
  }

  function zoomAt(factor, x = 500, y = 335) {
    const next = Math.max(minZoom, Math.min(maxZoom, state.zoom + factor));
    if (next === state.zoom) return;
    const [width, height] = viewport();
    const [centerX, centerY] = world([state.centerLon, state.centerLat]);
    const anchor = geographic([centerX + x - width / 2, centerY + y - height / 2]);
    const [anchorX, anchorY] = world(anchor, next);
    [state.centerLon, state.centerLat] = geographic([
      anchorX - x + width / 2, anchorY - y + height / 2,
    ], next);
    state.zoom = next;
    if (state.snapshot) renderMap();
  }

  function highlightCell(hexId) {
    for (const [id, polygon] of state.polygons) polygon.classList.toggle('selected', id === hexId);
    const selected = state.polygons.get(hexId);
    $('selectedLayer').replaceChildren();
    if (selected) $('selectedLayer').append(svgEl('polygon', {
      points: selected.getAttribute('points'), class: 'selected-halo',
    }));
  }

  function renderHotspots() {
    const container = $('hotspots');
    container.replaceChildren();
    const top = [...state.snapshot.cells].sort((a, b) => b.predicted_demand - a.predicted_demand).slice(0, 5);
    if (!top.length) { container.textContent = 'Chưa có ô H3 hợp lệ.'; return; }
    for (const [index, cell] of top.entries()) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'hotspot';
      const rank = document.createElement('span'); rank.className = 'rank'; rank.textContent = String(index + 1).padStart(2, '0');
      const body = document.createElement('span'); body.className = 'hotspot-main';
      const id = document.createElement('span'); id.className = 'hotspot-id'; id.textContent = cell.hex_id_7;
      const bar = document.createElement('span'); bar.className = 'hotspot-bar';
      const fill = document.createElement('i'); fill.style.width = (cell.predicted_demand / Math.max(top[0].predicted_demand, .01) * 100) + '%';
      bar.append(fill); body.append(id, bar);
      const value = document.createElement('span'); value.className = 'hotspot-value'; value.textContent = fmt(cell.predicted_demand);
      button.append(rank, body, value);
      button.addEventListener('click', () => selectCell(cell.hex_id_7));
      container.append(button);
    }
  }

  function renderChart(history, title) {
    $('chartTitle').textContent = title;
    const svg = $('trendChart');
    svg.replaceChildren();
    const list = history || [];
    $('chartValue').textContent = list.length ? fmt(list[list.length - 1].demand) : '—';
    if (list.length) {
      const unit = document.createElement('small'); unit.textContent = 'chuyến / 10 phút'; $('chartValue').append(unit);
    }
    $('chartFrom').textContent = list.length ? short(list[0].at) : '—';
    $('chartTo').textContent = list.length ? short(list[list.length - 1].at) : '—';
    if (!list.length) return;
    const ceiling = Math.max(1, ...list.map((item) => item.demand)) * 1.15;
    const points = list.map((item, i) => [
      6 + i * 388 / Math.max(list.length - 1, 1), 125 - item.demand / ceiling * 108,
    ]);
    for (const y of [24, 73, 124]) svg.append(svgEl('line', { x1: 0, x2: 400, y1: y, y2: y, stroke: '#36586b', 'stroke-dasharray': '3 6', opacity: '.75' }));
    const defs = svgEl('defs'); const gradient = svgEl('linearGradient', { id: 'trendFill', x1: '0', y1: '0', x2: '0', y2: '1' });
    gradient.append(svgEl('stop', { offset: '0%', 'stop-color': '#49e6d0', 'stop-opacity': '.34' }), svgEl('stop', { offset: '100%', 'stop-color': '#49e6d0', 'stop-opacity': '0' }));
    defs.append(gradient); svg.append(defs);
    const path = points.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
    const [first] = points; const last = points[points.length - 1];
    svg.append(svgEl('path', { d: `${path} L${last[0]},134 L${first[0]},134 Z`, fill: 'url(#trendFill)' }));
    svg.append(svgEl('path', { d: path, fill: 'none', stroke: '#57ead5', 'stroke-width': '2.8', 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
    svg.append(svgEl('circle', { cx: last[0], cy: last[1], r: '5', fill: '#7bffe1', stroke: '#0c2934', 'stroke-width': '3' }));
  }

  function resetDetail() {
    $('detailTitle').textContent = 'Chọn một ô trên bản đồ';
    $('detailText').textContent = 'Bấm vào ô H3 hoặc danh sách điểm nóng để xem dự báo 10 / 30 / 60 phút và lịch sử của ô đó.';
    $('horizonDetail').replaceChildren();
  }

  async function selectCell(hexId) {
    const snapshot = state.snapshot;
    if (!snapshot || !state.polygons.has(hexId)) return;
    state.selected = hexId;
    highlightCell(hexId);
    $('detailTitle').textContent = 'Ô H3 ' + hexId;
    $('detailText').textContent = 'Đang lấy lịch sử và ba khoảng dự báo…';
    $('horizonDetail').replaceChildren();
    const request = ++state.detailRequest;
    const params = new URLSearchParams({ travel_mode: state.mode, forecast_start_utc: snapshot.forecast_start_utc });
    try {
      const response = await fetch(`/v1/dashboard/hex/${encodeURIComponent(hexId)}?${params}`, { cache: 'no-store' });
      if (!response.ok) throw new Error('Không thể tải chi tiết ô H3.');
      const detail = await response.json();
      if (request !== state.detailRequest || snapshot !== state.snapshot) return;
      renderChart(detail.history, 'Ô ' + hexId);
      $('detailText').textContent = `Dự báo bắt đầu ${time(snapshot.forecast_start_utc)} · mỗi số là tổng chuyến trong khoảng tương ứng.`;
      const cards = $('horizonDetail');
      for (const minutes of [10, 30, 60]) {
        const card = document.createElement('div');
        const label = document.createElement('span'); label.textContent = minutes + ' PHÚT TỚI';
        const value = document.createElement('strong'); value.textContent = detail.horizons[String(minutes)] == null ? '—' : fmt(detail.horizons[String(minutes)]);
        card.append(label, value); cards.append(card);
      }
    } catch (error) {
      if (request !== state.detailRequest) return;
      $('detailText').textContent = error.message;
      renderChart(snapshot.region_history, 'Toàn vùng');
    }
  }

  $('modeButtons').addEventListener('click', (event) => {
    const button = event.target.closest('[data-mode]');
    if (!button || button.dataset.mode === state.mode) return;
    state.mode = button.dataset.mode; state.time = 'live'; state.selected = null; ++state.detailRequest;
    state.fitted = false;
    setActive('modeButtons', 'mode', state.mode); load();
  });
  $('horizonButtons').addEventListener('click', (event) => {
    const button = event.target.closest('[data-horizon]');
    if (!button || Number(button.dataset.horizon) === state.horizon) return;
    state.horizon = Number(button.dataset.horizon); state.time = 'live'; ++state.detailRequest;
    setActive('horizonButtons', 'horizon', state.horizon); load();
  });
  $('layerButtons').addEventListener('click', (event) => {
    const button = event.target.closest('[data-layer]');
    if (!button || button.dataset.layer === state.layer) return;
    state.layer = button.dataset.layer;
    setActive('layerButtons', 'layer', state.layer);
    if (state.snapshot) renderMap();
  });
  $('timeSelect').addEventListener('change', (event) => { state.time = event.target.value; ++state.detailRequest; load(); });
  $('refreshBtn').addEventListener('click', () => { state.seconds = 30; load(); });
  $('zoomIn').addEventListener('click', () => zoomAt(1, ...viewport().map((n) => n / 2)));
  $('zoomOut').addEventListener('click', () => zoomAt(-1, ...viewport().map((n) => n / 2)));
  $('zoomFit').addEventListener('click', resetMap);
  const opacityControl = $('hexOpacity');
  function updateOpacity() {
    const percentage = Number(opacityControl.value);
    $('mapStage').style.setProperty('--hex-opacity', String(percentage / 100));
    $('opacityValue').textContent = `${percentage}%`;
  }
  opacityControl.addEventListener('input', updateOpacity);
  updateOpacity();

  const map = $('hexMap');
  map.addEventListener('wheel', (event) => {
    event.preventDefault();
    const rect = map.getBoundingClientRect();
    zoomAt(event.deltaY < 0 ? 1 : -1,
      event.clientX - rect.left, event.clientY - rect.top);
  }, { passive: false });
  let drag = null;
  map.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;
    drag = {
      x: event.clientX, y: event.clientY,
      center: world([state.centerLon, state.centerLat]),
      hex: event.target.dataset.hex, moved: false,
    };
    map.setPointerCapture(event.pointerId); map.classList.add('dragging');
    $('mapTooltip').hidden = true;
  });
  map.addEventListener('pointermove', (event) => {
    if (!drag) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) < 4 && !drag.moved) return;
    drag.moved = true;
    [state.centerLon, state.centerLat] = geographic([
      drag.center[0] - dx, drag.center[1] - dy,
    ]);
    if (state.snapshot) renderMap();
  });
  const stopDrag = () => { drag = null; map.classList.remove('dragging'); };
  map.addEventListener('pointerup', () => {
    if (drag && !drag.moved && drag.hex) selectCell(drag.hex);
    stopDrag();
  });
  map.addEventListener('pointercancel', stopDrag);
  window.addEventListener('resize', () => { if (state.snapshot) renderMap(); });

  const tick = () => {
    $('clock').textContent = clockFormat.format(new Date()) + ' · ICT';
    state.seconds -= 1;
    if (state.seconds <= 0) {
      state.seconds = 30;
      if (state.time === 'live') load();
    }
    $('refreshCountdown').textContent = state.time === 'live' ? `${state.seconds}s` : 'tạm dừng';
    if (state.snapshot && state.seconds % 10 === 0) updateFreshness();
  };
  tick();
  setInterval(tick, 1000);
  load();
})();
