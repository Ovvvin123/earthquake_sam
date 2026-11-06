// Offline checks of owned gallery files and UI logic; does not open a browser.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const data = path.join(__dirname, 'Earth_dataset_v7');
const qa = path.join(data, 'qa');
const manifest = JSON.parse(fs.readFileSync(path.join(data, 'manifest.json'), 'utf8'));
const html = fs.readFileSync(path.join(qa, 'index.html'), 'utf8');
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(source); // Parse without a browser, network or UI automation.
const ids = ['split','query','prev','next','counter','detail','title','info','links','full','preview','cards'];
const elements = new Map();
function element(id = '') {
  return {id, value:'', textContent:'', innerHTML:'', children:[], dataset:{}, style:{},
    classList:{toggle(){}}, appendChild(c){this.children.push(c);},
    replaceChildren(){this.children=[];}, scrollIntoView(){}};
}
for (const id of ids) {
  assert(html.includes('id="'+id+'"'));
  elements.set(id, element(id));
}
elements.get('split').value='all';
const events={};
const context=vm.createContext({location:{hash:''},document:{
  getElementById:id=>{assert(elements.has(id), id);return elements.get(id);},
  createElement:()=>element(),
  querySelectorAll:selector=>{assert.equal(selector,'.card');return elements.get('cards').children;},
  addEventListener:(name,fn)=>events[name]=fn,
}});
vm.runInContext(source,context,{timeout:10000});
assert.equal(elements.get('counter').textContent,'1 / '+manifest.tiles.length);
elements.get('next').onclick();
assert.equal(elements.get('counter').textContent,'2 / '+manifest.tiles.length);
elements.get('prev').onclick();
elements.get('prev').onclick();
assert.equal(elements.get('counter').textContent,manifest.tiles.length+' / '+manifest.tiles.length);
events.keydown({target:{tagName:'DIV'},key:'ArrowRight'});
assert.equal(elements.get('counter').textContent,'1 / '+manifest.tiles.length);
for (const split of ['train','val','test']) {
  elements.get('split').value=split; elements.get('split').onchange();
  assert.equal(elements.get('cards').children.length,manifest.provenance.counts[split]);
  assert.equal(elements.get('counter').textContent,'1 / '+manifest.provenance.counts[split]);
}
elements.get('split').value='all';elements.get('split').onchange();
const last=manifest.tiles[manifest.tiles.length-1].key;
elements.get('query').value=last;elements.get('query').oninput();
assert.equal(elements.get('cards').children.length,1);
assert(elements.get('title').textContent.startsWith(last));
elements.get('query').value='__absent__';elements.get('query').oninput();
assert.equal(elements.get('detail').style.display,'none');
elements.get('query').value='';elements.get('query').oninput();
assert.equal(elements.get('cards').children.length,manifest.tiles.length);
let links=0;
for(const tile of manifest.tiles){
  for(const rel of [`qa/tiles/${tile.key}.png`,`qa/thumbs/${tile.key}.jpg`,
    `rasters/${tile.key}_elevation.tif`,`rasters/${tile.key}_masks.tif`]){
    assert(fs.statSync(path.join(data,rel)).size>0,rel);links++;
  }
}
for(const rel of ['qa/split_map.png','qa/elevation_legend.png','qa/validation_report.json',
  'qa/reader_validation.json','qa/geometry_audit.json','manifest.json','README.txt']){
  assert(fs.statSync(path.join(data,rel)).size>0,rel);links++;
}
assert(!/<(?:script|img)[^>]+(?:src)=["']https?:/i.test(html));
const result={status:'PASS',tiles:manifest.tiles.length,local_file_links_checked:links,
  javascript_syntax:'PASS',ui_logic_fixture:'PASS',checks:['next','previous','wraparound','arrow key',
  'train/val/test filtering','tile search','empty search result','reset'],
  browser_rendered:false,browser_limitation:'Built-in browser policy disallows file://; no browser workaround used.',
  external_script_or_image_dependencies:false};
fs.writeFileSync(path.join(qa,'gallery_validation.json'),JSON.stringify(result,null,2));
fs.copyFileSync(__filename,path.join(data,'provenance',path.basename(__filename)));
console.log(JSON.stringify(result,null,2));
