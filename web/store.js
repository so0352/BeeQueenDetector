// 保存した画像の記録（端末内の IndexedDB。どこにも送らない）。
// iPhone の Safari からは写真アプリへ直接保存できないため、いったんここに貯めて、
// 「記録」画面の共有ボタン（共有シート →「画像を保存」）で写真に移す。

const DB_NAME = 'bee-queen-detector';
const STORE = 'snaps';

let dbp = null;

function db() {
  if (!dbp) {
    dbp = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: 'id', autoIncrement: true });
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  return dbp;
}

async function tx(mode, fn) {
  const d = await db();
  return new Promise((resolve, reject) => {
    const t = d.transaction(STORE, mode);
    const r = fn(t.objectStore(STORE));
    t.oncomplete = () => resolve(r?.result);
    t.onerror = () => reject(t.error);
  });
}

/** @param {{time:number, blob:Blob, auto:boolean, conf:number}} rec */
export const add = (rec) => tx('readwrite', (s) => s.add(rec));
export const all = () => tx('readonly', (s) => s.getAll());
export const count = () => tx('readonly', (s) => s.count());
export const remove = (id) => tx('readwrite', (s) => s.delete(id));
export const clear = () => tx('readwrite', (s) => s.clear());
