import json
from pathlib import Path
from playwright.sync_api import sync_playwright
base='http://127.0.0.1:8840/'
result={};errors=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1440,'height':1000});page.on('pageerror',lambda e:errors.append(str(e)))
 for path in ['', 'A3_genea_atlas.html','A3_gesture_benchmark.html','A3_speech_demo.html']:
  response=page.goto(base+path,wait_until='domcontentloaded');page.wait_for_timeout(350)
  assert response.status==200,(path,response.status)
  result[path or '/']={'status':response.status,'title':page.title(),'h1':page.locator('h1').first.inner_text()}
 page.goto(base+'#review',wait_until='domcontentloaded');page.wait_for_selector('#qlist .qi')
 assert page.locator('#q-count').inner_text()=='529 个片段'
 page.select_option('#f-kind','atom');page.select_option('#f-grade','A')
 assert page.locator('#q-count').inner_text()=='215 个片段'
 page.locator('#qlist .qi').first.click();page.wait_for_timeout(200)
 before=page.locator('#scrub').input_value();page.wait_for_timeout(300);after=page.locator('#scrub').input_value()
 assert before!=after,(before,after)
 page.locator('.decision button[data-v=accept]').click();page.fill('#r-note','UI verification only; ephemeral browser context');page.click('#r-save')
 with page.expect_download() as download:page.click('#btn-export')
 result['review']={'count':529,'A_atoms':215,'animation_advances':True,'export':download.value.suggested_filename}
 page.screenshot(path='results/execution_20261027/review_desktop.png')
 page.goto(base+'A3_genea_atlas.html',wait_until='domcontentloaded')
 page.locator('[data-rec]').first.click();page.wait_for_timeout(100)
 assert page.locator('#cards .card').count()>0
 result['atlas']={'source_filter_cards':page.locator('#cards .card').count(),'matrix_rows':page.locator('#semantic-style tr').count()}
 page.goto(base+'A3_gesture_benchmark.html',wait_until='domcontentloaded')
 assert page.locator('#sonic-rows tr').count()==20
 result['benchmark']={'sonic_rows':20}
 page.goto(base+'A3_speech_demo.html',wait_until='domcontentloaded')
 page.wait_for_function('document.querySelector("video").readyState >= 1')
 duration=page.locator('video').evaluate('(v)=>v.duration');assert 20<=duration<=30
 page.locator('#events button').nth(1).click();page.wait_for_timeout(600)
 state=page.locator('video').evaluate('(v)=>({t:v.currentTime,paused:v.paused,error:v.error?.message})')
 assert state['t']>6 and not state['paused'] and not state.get('error'),state
 result['demo']={'duration':duration,'playback':state}
 page.screenshot(path='results/execution_20261027/demo_desktop.png')
 page.set_viewport_size({'width':390,'height':844})
 for path in ['A3_genea_review.html','A3_genea_atlas.html','A3_speech_demo.html']:
  page.goto(base+path,wait_until='domcontentloaded');page.wait_for_timeout(200)
  size=page.evaluate('({w:innerWidth,scroll:document.documentElement.scrollWidth})')
  result.setdefault(path,{})['mobile'] = size
  assert size['scroll']<=size['w']+2,(path,size)
 page.screenshot(path='results/execution_20261027/demo_mobile.png')
 browser.close()
result['page_errors']=errors
Path('results/execution_20261027/ui_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
assert not errors,errors
print(json.dumps(result,ensure_ascii=False,indent=2))
