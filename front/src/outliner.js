// Left panel: Bodies / Joints / Cameras tree, "/" in ids shown as collapsible folders.
import { app, clickSelect } from './state.js';
import { LIST_KEY } from './model.js';

const root = document.getElementById('outliner-tree');
const collapsed = new Set(); // {"kind:folder/path/"} collapsed folders ("kind:" = whole section)

const SHAPE_ICONS = { box: '▣', sphere: '●', cylinder: '◖', plane: '▭' }; // {shape type: icon}
const JOINT_ICONS = { revolute: '⟳', prismatic: '⇕', fixed: '⊕' }; // {joint type: icon}
const SECTIONS = [['body', 'Bodies'], ['joint', 'Joints'], ['camera', 'Cameras']]; // [[kind, title]]

// one tree line: [caret] [icon] name ... meta
function row(depth, className, caret, icon, name, meta) {
  const el = document.createElement('div');
  el.className = `tree-row ${className}`;
  el.style.paddingLeft = `${8 + depth * 14}px`;
  for (const [cls, text] of [['caret', caret], ['icon', icon], ['name', name], ['meta', meta]]) {
    const span = document.createElement('span');
    span.className = cls;
    span.textContent = text;
    el.append(span);
  }
  root.append(el);
  return el;
}

export function render() {
  root.innerHTML = '';
  const scene = app.shown;
  if (!scene) return;
  const selected = new Set(app.selection.map(s => `${s.kind}:${s.id}`)); // {"kind:id"}

  for (const [kind, title] of SECTIONS) {
    const items = scene[LIST_KEY[kind]];
    // folder tree from "/"-separated ids
    const tree = { folders: new Map(), items: [] }; // {folders: {name: node}, items: [item]}
    for (const item of items) {
      let node = tree;
      for (const part of item.id.split('/').slice(0, -1)) {
        if (!node.folders.has(part)) node.folders.set(part, { folders: new Map(), items: [] });
        node = node.folders.get(part);
      }
      node.items.push(item);
    }

    // draw a folder node's children; path = "kind:a/b/"
    const draw = (node, path, depth) => {
      for (const [name, child] of node.folders) {
        const childPath = `${path}${name}/`;
        const isOpen = !collapsed.has(childPath);
        const el = row(depth, 'folder', isOpen ? '▼' : '▶', '▤', name, child.items.length + child.folders.size);
        el.onclick = () => {
          if (isOpen) collapsed.add(childPath);
          else collapsed.delete(childPath);
          render();
        };
        if (isOpen) draw(child, childPath, depth + 1);
      }
      for (const item of node.items) {
        const icon = kind === 'body' ? SHAPE_ICONS[item.shape.type] : kind === 'joint' ? JOINT_ICONS[item.type] : '◫';
        const meta = kind === 'body' ? (item.static ? 'static' : '') : kind === 'joint' ? item.type : item.body;
        const el = row(depth, selected.has(`${kind}:${item.id}`) ? 'selected' : '', '', icon, item.id.split('/').at(-1), meta);
        if (kind === 'body') el.querySelector('.icon').style.color = item.color;
        el.title = item.id;
        el.onclick = (e) => clickSelect({ kind, id: item.id }, e.shiftKey);
      }
    };

    const sectionPath = `${kind}:`;
    const isOpen = !collapsed.has(sectionPath);
    const header = row(0, 'header', isOpen ? '▼' : '▶', '', title, items.length);
    header.onclick = () => {
      if (isOpen) collapsed.add(sectionPath);
      else collapsed.delete(sectionPath);
      render();
    };
    if (isOpen) draw(tree, sectionPath, 1);
  }
}
