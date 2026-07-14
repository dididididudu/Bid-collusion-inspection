(function () {
  "use strict";

  const COLS = 10;
  const ROWS = 20;
  const BLOCK = 30;
  const PREVIEW = 28;
  const LOCK_DELAY = 450;
  const DAS = 140;
  const ARR = 42;
  const STORAGE_KEY = "codex-tetris-best";

  const COLORS = {
    I: "#28d7e8",
    O: "#ffd34e",
    T: "#b46cff",
    S: "#4ddb76",
    Z: "#ff5e6c",
    J: "#5b8cff",
    L: "#ff9f43",
    GHOST: "rgba(255,255,255,0.18)",
  };

  const SHAPES = {
    I: [
      [0, 0],
      [-1, 0],
      [1, 0],
      [2, 0],
    ],
    O: [
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
    ],
    T: [
      [0, 0],
      [-1, 0],
      [1, 0],
      [0, -1],
    ],
    S: [
      [0, 0],
      [1, 0],
      [0, -1],
      [-1, -1],
    ],
    Z: [
      [0, 0],
      [-1, 0],
      [0, -1],
      [1, -1],
    ],
    J: [
      [0, 0],
      [-1, 0],
      [1, 0],
      [-1, -1],
    ],
    L: [
      [0, 0],
      [-1, 0],
      [1, 0],
      [1, -1],
    ],
  };

  const KICKS = [
    [0, 0],
    [-1, 0],
    [1, 0],
    [0, -1],
    [-2, 0],
    [2, 0],
  ];

  class PieceBag {
    constructor() {
      this.queue = [];
    }

    next() {
      if (this.queue.length < 7) this.queue.push(...this.shuffle());
      return this.queue.shift();
    }

    peek() {
      if (this.queue.length < 7) this.queue.push(...this.shuffle());
      return this.queue[0];
    }

    shuffle() {
      const pieces = Object.keys(SHAPES);
      for (let i = pieces.length - 1; i > 0; i -= 1) {
        const j = Math.floor(Math.random() * (i + 1));
        [pieces[i], pieces[j]] = [pieces[j], pieces[i]];
      }
      return pieces;
    }
  }

  class TetrisEngine {
    constructor() {
      this.best = Number(localStorage.getItem(STORAGE_KEY) || 0);
      this.reset();
    }

    reset() {
      this.board = Array.from({ length: ROWS }, () => Array(COLS).fill(null));
      this.bag = new PieceBag();
      this.score = 0;
      this.lines = 0;
      this.level = 1;
      this.holdType = null;
      this.holdLocked = false;
      this.paused = false;
      this.gameOver = false;
      this.lockTimer = 0;
      this.spawn();
    }

    spawn(type = this.bag.next()) {
      this.active = { type, x: 4, y: 1, rotation: 0 };
      this.holdLocked = false;
      this.lockTimer = 0;
      if (this.collides(this.active)) {
        this.gameOver = true;
        this.saveBest();
      }
    }

    cells(piece = this.active) {
      return SHAPES[piece.type].map(([x, y]) => {
        let rx = x;
        let ry = y;
        for (let i = 0; i < piece.rotation; i += 1) {
          [rx, ry] = [-ry, rx];
        }
        return [piece.x + rx, piece.y + ry];
      });
    }

    collides(piece) {
      return this.cells(piece).some(([x, y]) => {
        return x < 0 || x >= COLS || y >= ROWS || (y >= 0 && this.board[y][x]);
      });
    }

    move(dx, dy) {
      if (this.paused || this.gameOver) return false;
      const moved = { ...this.active, x: this.active.x + dx, y: this.active.y + dy };
      if (this.collides(moved)) return false;
      this.active = moved;
      if (dy === 0) this.lockTimer = 0;
      return true;
    }

    rotate(dir) {
      if (this.paused || this.gameOver || this.active.type === "O") return false;
      const rotation = (this.active.rotation + dir + 4) % 4;
      for (const [kx, ky] of KICKS) {
        const rotated = { ...this.active, rotation, x: this.active.x + kx, y: this.active.y + ky };
        if (!this.collides(rotated)) {
          this.active = rotated;
          this.lockTimer = 0;
          return true;
        }
      }
      return false;
    }

    softDrop() {
      if (this.move(0, 1)) {
        this.score += 1;
        return true;
      }
      return false;
    }

    hardDrop() {
      if (this.paused || this.gameOver) return;
      let distance = 0;
      while (this.move(0, 1)) distance += 1;
      this.score += distance * 2;
      this.lock();
    }

    hold() {
      if (this.paused || this.gameOver || this.holdLocked) return;
      const current = this.active.type;
      if (this.holdType) {
        const next = this.holdType;
        this.holdType = current;
        this.active = { type: next, x: 4, y: 1, rotation: 0 };
        if (this.collides(this.active)) {
          this.gameOver = true;
          this.saveBest();
        }
      } else {
        this.holdType = current;
        this.spawn();
      }
      this.holdLocked = true;
    }

    tick(delta) {
      if (this.paused || this.gameOver) return;
      this.gravityTimer = (this.gravityTimer || 0) + delta;
      const interval = Math.max(85, 820 - (this.level - 1) * 62);
      while (this.gravityTimer >= interval) {
        this.gravityTimer -= interval;
        if (!this.move(0, 1)) this.lockTimer += interval;
      }
      if (this.onGround()) {
        this.lockTimer += delta;
        if (this.lockTimer >= LOCK_DELAY) this.lock();
      } else {
        this.lockTimer = 0;
      }
    }

    onGround() {
      return this.collides({ ...this.active, y: this.active.y + 1 });
    }

    ghost() {
      const ghost = { ...this.active };
      while (!this.collides({ ...ghost, y: ghost.y + 1 })) ghost.y += 1;
      return ghost;
    }

    lock() {
      for (const [x, y] of this.cells()) {
        if (y < 0) {
          this.gameOver = true;
          this.saveBest();
          return;
        }
        this.board[y][x] = this.active.type;
      }
      const cleared = this.clearLines();
      if (cleared > 0) {
        const table = [0, 100, 300, 500, 800];
        this.lines += cleared;
        this.level = Math.floor(this.lines / 10) + 1;
        this.score += table[cleared] * this.level;
      }
      this.spawn();
      this.saveBest();
    }

    clearLines() {
      const kept = this.board.filter((row) => row.some((cell) => !cell));
      const cleared = ROWS - kept.length;
      while (kept.length < ROWS) kept.unshift(Array(COLS).fill(null));
      this.board = kept;
      return cleared;
    }

    togglePause() {
      if (!this.gameOver) this.paused = !this.paused;
    }

    saveBest() {
      if (this.score > this.best) {
        this.best = this.score;
        localStorage.setItem(STORAGE_KEY, String(this.best));
      }
    }
  }

  const boardCanvas = document.getElementById("board");
  const boardCtx = boardCanvas.getContext("2d");
  const nextCtx = document.getElementById("next").getContext("2d");
  const holdCtx = document.getElementById("hold").getContext("2d");
  const overlay = document.getElementById("pauseOverlay");
  const overlayTitle = document.getElementById("overlayTitle");
  const overlaySubtitle = document.getElementById("overlaySubtitle");
  const scoreEl = document.getElementById("score");
  const levelEl = document.getElementById("level");
  const linesEl = document.getElementById("lines");
  const bestEl = document.getElementById("best");
  const restartBtn = document.getElementById("restartBtn");
  const engine = new TetrisEngine();
  const keys = new Map();
  let lastTime = performance.now();

  function drawCell(ctx, x, y, size, color, alpha = 1) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    ctx.fillRect(x * size + 1, y * size + 1, size - 2, size - 2);
    ctx.fillStyle = "rgba(255,255,255,0.22)";
    ctx.fillRect(x * size + 3, y * size + 3, size - 6, 4);
    ctx.restore();
  }

  function drawBoard() {
    boardCtx.clearRect(0, 0, boardCanvas.width, boardCanvas.height);
    boardCtx.fillStyle = "#0b0e14";
    boardCtx.fillRect(0, 0, boardCanvas.width, boardCanvas.height);

    boardCtx.strokeStyle = "#262d3b";
    boardCtx.lineWidth = 1;
    for (let x = 0; x <= COLS; x += 1) {
      boardCtx.beginPath();
      boardCtx.moveTo(x * BLOCK + 0.5, 0);
      boardCtx.lineTo(x * BLOCK + 0.5, ROWS * BLOCK);
      boardCtx.stroke();
    }
    for (let y = 0; y <= ROWS; y += 1) {
      boardCtx.beginPath();
      boardCtx.moveTo(0, y * BLOCK + 0.5);
      boardCtx.lineTo(COLS * BLOCK, y * BLOCK + 0.5);
      boardCtx.stroke();
    }

    engine.board.forEach((row, y) => {
      row.forEach((type, x) => {
        if (type) drawCell(boardCtx, x, y, BLOCK, COLORS[type]);
      });
    });

    for (const [x, y] of engine.cells(engine.ghost())) {
      if (y >= 0) drawCell(boardCtx, x, y, BLOCK, COLORS.GHOST, 1);
    }

    for (const [x, y] of engine.cells()) {
      if (y >= 0) drawCell(boardCtx, x, y, BLOCK, COLORS[engine.active.type]);
    }
  }

  function drawPreview(ctx, type) {
    ctx.clearRect(0, 0, 112, 112);
    ctx.fillStyle = "#10151f";
    ctx.fillRect(0, 0, 112, 112);
    if (!type) return;
    const cells = SHAPES[type];
    const xs = cells.map(([x]) => x);
    const ys = cells.map(([, y]) => y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const offsetX = Math.floor((4 - (maxX - minX + 1)) / 2) - minX;
    const offsetY = Math.floor((4 - (maxY - minY + 1)) / 2) - minY;
    cells.forEach(([x, y]) => drawCell(ctx, x + offsetX, y + offsetY, PREVIEW, COLORS[type]));
  }

  function syncHud() {
    scoreEl.textContent = engine.score.toLocaleString("zh-CN");
    levelEl.textContent = String(engine.level);
    linesEl.textContent = String(engine.lines);
    bestEl.textContent = engine.best.toLocaleString("zh-CN");
    drawPreview(nextCtx, engine.bag.peek());
    drawPreview(holdCtx, engine.holdType);

    overlay.classList.toggle("hidden", !engine.paused && !engine.gameOver);
    if (engine.gameOver) {
      overlayTitle.textContent = "游戏结束";
      overlaySubtitle.textContent = "按 R 或点击右侧按钮重新开始";
    } else {
      overlayTitle.textContent = "暂停";
      overlaySubtitle.textContent = "按 P 或 Esc 继续";
    }
  }

  function repeatAction(code, action, now) {
    const state = keys.get(code);
    if (!state) return;
    if (!state.fired) {
      action();
      state.fired = true;
      state.next = now + DAS;
      return;
    }
    if (now >= state.next) {
      action();
      state.next = now + ARR;
    }
  }

  function updateInput(now) {
    repeatAction("ArrowLeft", () => engine.move(-1, 0), now);
    repeatAction("ArrowRight", () => engine.move(1, 0), now);
    repeatAction("ArrowDown", () => engine.softDrop(), now);
  }

  function loop(now) {
    const delta = Math.min(50, now - lastTime);
    lastTime = now;
    updateInput(now);
    engine.tick(delta);
    drawBoard();
    syncHud();
    requestAnimationFrame(loop);
  }

  window.addEventListener("keydown", (event) => {
    const repeatable = ["ArrowLeft", "ArrowRight", "ArrowDown"];
    if (repeatable.includes(event.code)) {
      event.preventDefault();
      if (!keys.has(event.code)) keys.set(event.code, { fired: false, next: 0 });
      return;
    }

    if (["Space", "ArrowUp", "KeyX", "KeyZ", "KeyC", "ShiftLeft", "ShiftRight", "KeyP", "Escape", "KeyR"].includes(event.code)) {
      event.preventDefault();
    }

    if (event.repeat) return;
    if (event.code === "Space") engine.hardDrop();
    if (event.code === "ArrowUp" || event.code === "KeyX") engine.rotate(1);
    if (event.code === "KeyZ") engine.rotate(-1);
    if (event.code === "KeyC" || event.code === "ShiftLeft" || event.code === "ShiftRight") engine.hold();
    if (event.code === "KeyP" || event.code === "Escape") engine.togglePause();
    if (event.code === "KeyR") engine.reset();
  });

  window.addEventListener("keyup", (event) => {
    keys.delete(event.code);
  });

  restartBtn.addEventListener("click", () => engine.reset());
  requestAnimationFrame(loop);
})();
