"""Build four figures and a self-contained HTML report from measured data only."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import html
import json
import math
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from client import percentile,summarize
R=ROOT/'results';F=ROOT/'figures';F.mkdir(exist_ok=True)

def read_lines(name):
    p=R/name
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
def load(name):
    p=R/name;return json.loads(p.read_text()) if p.exists() else {}
rows=read_lines('requests.jsonl');events=read_lines('events.jsonl');timeline=read_lines('timeline.jsonl')
groups=defaultdict(list)
for row in rows:groups[row['run']].append(row)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'svg.fonttype':'none','figure.facecolor':'#fafbfe','axes.facecolor':'#fafbfe'})
colors=['#4f46e5','#e76f51','#2a9d8f','#d4a72c']

def ready(p):return any(c['type']=='Ready' and c['status']=='True' for c in p.get('status',{}).get('conditions',[]))
def save(fig,num):
    fig.tight_layout();fig.savefig(F/f'{num}.svg',bbox_inches='tight');svg=F/f'{num}.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n');fig.savefig(F/f'{num}.png',dpi=130,bbox_inches='tight');plt.close(fig)

def empty(ax):ax.text(.5,.5,'Нет завершённого измерения',ha='center',va='center',transform=ax.transAxes)

# 1: actual Ready and EndpointSlice trajectory, with every A attempt.
a=[r for r in rows if r['run'].startswith('A-')]
fig,ax=plt.subplots(2,1,figsize=(11,6),sharex=True)
if a:
    zero=min(r['sent_unix'] for r in a);last=max(r['sent_unix']+r['duration_seconds'] for r in a)
    tt=[t for t in timeline if zero-2<=t['unix']<=last+5]
    ax[0].step([(t['unix']-zero)/60 for t in tt],[sum(ready(p) for p in t['pods']) for t in tt],where='post',label='Ready Pods',color=colors[0])
    ax[0].step([(t['unix']-zero)/60 for t in tt],[sum(e.get('conditions',{}).get('ready',False) for item in t['endpoints'].get('items',[]) for e in item.get('endpoints',[])) for t in tt],where='post',label='Ready Service endpoints',color=colors[1],linestyle='--')
    for outcome in sorted(set(r['outcome'] for r in a)):
        rr=[r for r in a if r['outcome']==outcome]
        ax[1].scatter([(r['sent_unix']-zero)/60 for r in rr],[r['duration_seconds'] for r in rr],s=8,label=outcome,alpha=.6)
    for e in events:
        if e['event'] in ('rollout_B_begin','rollback_A_begin','forced_kill'):
            x=(e['unix']-zero)/60
            for axis in ax:axis.axvline(x,color='#8b93a7',ls=':',alpha=.7)
            ax[0].annotate(e['event'],(x,2),rotation=90,fontsize=8)
    # Shade gaps where only setup or archived attempts occurred.
    intervals=sorted((r['sent_unix'],r['sent_unix']+r['duration_seconds']) for r in a)
    end=intervals[0][1]
    for begin,finish in intervals[1:]:
        if begin-end>15:
            for axis in ax:axis.axvspan((end-zero)/60,(begin-zero)/60,color='#a0a8b8',alpha=.12)
        end=max(end,finish)
    ax[0].legend(loc='upper left');ax[1].legend();ax[0].set_ylim(0,4);ax[0].set_yticks(range(5))
else:empty(ax[0])
ax[0].set_ylabel('Количество');ax[1].set_ylabel('Длительность, с');ax[1].set_xlabel('Минуты от первого запроса A')
fig.suptitle('A · Прогрев, штатный rollout, откат и авария · CPU simulated');save(fig,1)

# 2: one actual ClientIP skew trace, including local queue and sampled gauges.
selected=groups.get('B-api-sticky-skew-1',[])
fig,ax=plt.subplots(2,1,figsize=(11,6),sharex=True)
if selected:
    zero=min(r['sent_unix'] for r in selected);last=max(r['sent_unix']+r['duration_seconds'] for r in selected)
    for i,pod in enumerate(sorted(set(r['pod'] for r in selected if r['pod']))):
        good=[r for r in selected if r['pod']==pod and r['outcome']=='completed']
        for axis,key,label in ((ax[0],'ttfc_seconds','TTFC'),(ax[1],'queue_seconds','Очередь')):
            axis.scatter([r['sent_unix']-zero for r in good],[r[key] for r in good],s=7,alpha=.5,label=pod,color=colors[i%4])
    for axis in ax:axis.legend(fontsize=8)
else:empty(ax[0])
ax[0].set_ylabel('TTFC, с');ax[1].set_ylabel('Время в очереди, с');ax[1].set_xlabel('Время отправки от начала прогона, с')
fig.suptitle('B · ClientIP, 80% через одного клиента · 12 запросов/с, 2 слота/Pod');save(fig,2)

# 3: all three repetitions, never pool pod quantiles.
fig,ax=plt.subplots(1,2,figsize=(11,5))
labels=[]
for panel,traffic in enumerate(('uniform','skew')):
    for j,policy in enumerate(('api','api-sticky')):
        ps=[];errs=[];reps=[]
        for rep in range(1,4):
            rr=groups.get(f'B-{policy}-{traffic}-{rep}',[])
            if rr:
                s=summarize(rr);reps.append(rep);ps.append(s['ttfc_p95']);errs.append(s['error_fraction'])
        if reps:
            ax[panel].plot(reps,ps,'o-',label=policy,color=colors[j])
            for x,y,error in zip(reps,ps,errs):ax[panel].annotate(f'err={error:.1%}',(x,y),xytext=(-4 if x==3 else 4,9),ha='right' if x==3 else 'left',textcoords='offset points',fontsize=8)
    ax[panel].set_title('Равномерно' if traffic=='uniform' else '80% через client-0')
    ax[panel].set_xlabel('Повтор, фаза 120 с');ax[panel].set_xticks([1,2,3]);ax[panel].legend()
ax[0].set_ylabel('Клиентский TTFC p95 успешных ответов, с')
fig.suptitle('B · Одинаковые расписания и входы; кеш пуст перед каждым прогоном');save(fig,3)

# 4: HPA trajectory and offered load, with actual CPU when available.
c=[r for r in rows if r['run'].startswith('C-')]
fig,ax=plt.subplots(3,1,figsize=(11,8),sharex=True)
if c:
    zero=min(r['sent_unix'] for r in c);last=max(r['sent_unix']+r['duration_seconds'] for r in c)+80
    tt=[t for t in timeline if zero-10<=t['unix']<=last]
    x=[(t['unix']-zero)/60 for t in tt]
    ax[0].step(x,[sum(ready(p) for p in t['pods']) for t in tt],where='post',label='Ready',color=colors[0])
    ax[0].step(x,[t['hpa'].get('items',[{}])[0].get('status',{}).get('desiredReplicas',2) if t['hpa'].get('items') else 2 for t in tt],where='post',label='HPA desired',ls='--',color=colors[1])
    bins=np.arange(zero,last+5,5)
    counts,edges=np.histogram([r['sent_unix'] for r in c],bins=bins)
    ax[1].step((edges[:-1]-zero)/60,counts/5,where='post',color=colors[2])
    cpu=load('prometheus/range-cpu.json')
    for i,series in enumerate(cpu.get('data',{}).get('result',[])):
        vals=[(float(t),float(v)) for t,v in series['values'] if zero<=float(t)<=last]
        if vals:ax[2].plot([(t-zero)/60 for t,v in vals],[v for t,v in vals],label=series['metric'].get('pod'),color=colors[i%4])
    ax[0].legend();ax[2].legend(fontsize=7)
else:empty(ax[0])
ax[0].set_ylabel('Реплики');ax[0].set_ylim(0,4);ax[0].set_yticks(range(5));ax[1].set_ylabel('Отправлено, запросов/с');ax[2].set_ylabel('CPU, ядра');ax[2].set_xlabel('Минуты от начала нагрузки C')
fig.suptitle('C · HPA CPU 55% от request=200m; min=2 max=3; downscale window=60 с');save(fig,4)

summaries={name:summarize(rr) for name,rr in groups.items()}
(R/'summary.json').write_text(json.dumps(summaries,indent=2,ensure_ascii=False)+'\n')

def esc(x):return html.escape(str(x))
def fmt(x):return '—' if x is None else f'{x:.3f}'
def table(names):
    out='<table><tr><th>Прогон</th><th>N успешно / всего</th><th>TTFC p50 / p95, с</th><th>Полный p95, с</th><th>Ошибки</th><th>Тайм-ауты</th><th>Лаг отправки p95, с</th></tr>'
    for name in names:
        s=summaries[name]
        out+=f'<tr><td>{esc(name)}</td><td>{s["observations"]} / {s["planned"]}</td><td>{fmt(s["ttfc_p50"])} / {fmt(s["ttfc_p95"])}</td><td>{fmt(s["duration_p95"])}</td><td>{s["error_fraction"]:.2%}</td><td>{s["timeout_fraction"]:.2%}</td><td>{fmt(s["schedule_lag_p95"])}</td></tr>'
    return out+'</table>'

def image(i):
    text=(F/f'{i}.svg').read_text();return text[text.index('<svg'):]

def pod_table():
    out='<table><tr><th>Прогон / Pod</th><th>N</th><th>TTFC p95, с</th><th>Очередь p95, с</th></tr>'
    for name,s in summaries.items():
        if not name.startswith('B-'):continue
        for pod,stats in s['pods'].items():
            out+=f'<tr><td>{esc(name)} / {esc(pod)}</td><td>{stats["n"]}</td><td>{fmt(stats["ttfc_p95"])}</td><td>{fmt(stats["queue_p95"])}</td></tr>'
    return out+'</table>'

def hpa_details():
    result=[]
    for t in timeline:
        for pod in t['pods']:
            name=pod['metadata']['name']
            if not any(e['event']=='load_start' and e.get('run')=='C-hpa-high' and e['unix']<=t['unix'] for e in events):continue
            if name in {x['pod'] for x in result}:continue
            if ready(pod):
                created=pod['metadata']['creationTimestamp']
                cond=next(c for c in pod['status']['conditions'] if c['type']=='Ready')
                result.append(dict(pod=name,created=created,ready=cond['lastTransitionTime']))
    (R/'hpa-pod-timings.json').write_text(json.dumps(result,indent=2)+'\n')
    return esc(json.dumps(result,indent=2,ensure_ascii=False))

net=load('network.json');clients=load('clients.json');env=load('environment-after.json') or load('environment-before.json');hist=load('histogram-comparison.json')
inventory_size=env.get('node_size',{}).get('stdout','не снят').strip()
node_storage=env.get('node_storage_KiB',{}).get('stdout','')
node_bytes=sum(int(line.split()[0])*1024 for line in node_storage.splitlines() if line.split() and line.split()[0].isdigit())
clients_text=', '.join(f"{p['metadata']['name']} = {p.get('status',{}).get('podIP')}" for p in clients.get('items',[]))
alerts=read_lines('alerts.jsonl');states=defaultdict(set)
for snap in alerts:
    for a in snap.get('data',{}).get('data',{}).get('alerts',[]):states[a['labels']['alertname']].add(a['state'])
# Preserve a counterexample if uniform ClientIP mapping happened to balance.
counter=[]
for name,s in summaries.items():
    if (name.startswith('B-api-sticky-uniform') or name.startswith('B-counterexample-uniform')) and len(s['pods'])==2:
        n=sorted(v['n'] for v in s['pods'].values())
        if n[-1]/sum(n)<.6:counter.append(name)

done={e['event'] for e in events}
status='Все три серии опытов завершены.' if {'A_complete','B_complete','C_complete'}<=done else 'Текущий статус: '+', '.join(f'{part}: '+('завершён' if part+'_complete' in done else 'ещё не завершён') for part in 'ABC')+'.'
body=f'''<p class="eyebrow">МГТУ · MLOPS · ДЗ 11 · CPU-стенд</p>
<h1>Сервис работает.<br>Пользователи ждут.</h1>
<p class="status">{esc(status)}</p><p class="lead">Управляемый backend, проверяемые Kubernetes-опыты и исходные измерения. Режим всех ответов: <code>backend_mode=simulated</code>.</p>
<p>Сформировано {datetime.now(timezone.utc).isoformat()}. Источник данных: <a href="results/requests.jsonl">requests.jsonl</a>, <a href="results/timeline.jsonl">Ready / EndpointSlice / HPA</a>, <a href="results/events.jsonl">события</a>, <a href="results/summary.json">сводка</a>.</p>
<h2>1. Стенд и воспроизводимость</h2>
<p>Продолжен backend-контракт ДЗ 8–10: <code>SimulatedBackend(StubBackend)</code>, фактически загруженный model_id, <code>verify_artifact</code> и SHA-256 каждого файла из manifest. Release gate предыдущего ДЗ не заменяется: A и B здесь — отдельные малые артефакты имитатора, а не новые реальные веса. Образ содержит обе неизменяемые версии; источник доступен всем Pod независимо от локального тома. ConfigMap выбирает версию при старте процесса; checksum values создаёт новую ревизию Deployment. Хеши server.py и backend.py в запущенных Pod сравнены с исходниками сдачи: <a href="results/runtime-source-hashes.json">runtime-source-hashes.json</a>. Локальные веса, torch и MLflow этому опыту не нужны.</p>
<p>kind 0.27.0, Kubernetes 1.32.2, Helm 3.17.3, Python 3.12.10 в контейнере. Точные Python-зависимости: <a href="requirements-runtime.lock">runtime lock</a> и <a href="uv.lock">uv.lock</a>. Базовый Python закреплён digest в Dockerfile; фактические imageID всех Pod сохранены в timeline и inventory. CPU/RAM Docker: 14 CPU / 8 318 976 000 байт (7,75 GiB). До создания стенда <code>df -h .</code> показывал 51 GiB свободного места; подробный снимок после подготовки: <a href="results/environment-before.json">environment-before</a>; итоговый: <a href="results/environment-after.json">environment-after</a>.</p>
<p>Deployment: 2 реплики, request 200m CPU / 96Mi, limit 1 CPU / 192Mi; rollout maxUnavailable=0, maxSurge=1. Прогрев 20 с, startup budget 120 с, readiness 1 с, liveness проверяет только живость HTTP. preStop: сброс readiness и 5 с продолжения HTTP для обновления сетевых правил; HTTP shutdown до 40 с, общий grace 50 с. Размеры томов: Prometheus emptyDir 512Mi (retention 2h / 256MB), Grafana 128Mi. Данные стенда удаляются вместе с отдельным kind-кластером; образ и результаты удаляются отдельными адресными командами из README.</p>
<p>Настройки backend для B/C: 2 слота, очередь 24, timeout очереди 15 с; LRU 16 префиксов; hit=0,04 с CPU, miss=0,25 с CPU, inter-chunk delay=0,08 с, 4 части. CPU создаётся вычислением SHA-256 с короткими уступками event loop, чтобы probes обслуживались. Фактический writable size контейнера kind по docker inspect --size: {esc(inventory_size)} байт. Фактические данные containerd + kubelet + журналов внутри узла: {node_bytes/2**30:.2f} GiB (<a href="results/environment-after.json">du по непересекающимся каталогам</a>); слои базового kind-образа учитываются отдельно в docker image inspect. Основные B/C выполнялись на v1, итоговый A — на v2 с исправлением graceful drain и классификации текстовых HTTP-ошибок клиента. Код backend.py и его параметры неизменны; v1 сохранена в results/source-v1 и runtime-source-hashes-v1.json. Это модель нагрузки имитатора; задержки не являются скоростью настоящей LLM. Гарантия отсутствия обрывов относится к записанной нагрузке и тайм-аутам этого прогона; при изменении лимитов длительности и очереди бюджет graceful shutdown нужно пересчитать. Для штатного rollout interval увеличен до 0,25 с, длинный поток 128 частей длится около 32 с и пересекает замену старого Pod.</p>
<h2>2. Опыт A: прогрев, выпуск, откат, авария</h2>
<p><strong>Гипотеза:</strong> доступность HTTP не доказывает готовность backend. В исходном дефекте <code>brokenProbes=true</code> startup/readiness отвечают 200 до завершения прогрева. Результат — рабочий Service включает холодный Pod и приложение возвращает 503. Исправление: startup/readiness проверяют завершённый прогрев, загрузку и отсутствие draining; liveness остаётся отдельной проверкой процесса. Прогоны до/после содержат по 225 запланированных запросов при 3 запросах/с в течение 75 с. В journal сохранены <code>received warm=false</code>, а в timeline — Ready и условия EndpointSlice. Исправленный опыт проверяет отсутствие таких рабочих запросов к новым Pod.</p>
{table([n for n in summaries if n.startswith('A-')])}
<p>При выпуске B открыты потоки непосредственно к обеим старым репликам, затем начинается rolling update. Фоновые запросы идут через настоящий ClusterIP из bridge Pod; host port-forward к Service не используется как балансировщик. Проверяется завершение каждого потока и каждого фонового запроса. На графике серым выделены промежутки подготовки и архивных неуспешных проверок. Для заведомо неготовой B <code>neverReady=true</code> делает прогрев неуспешным: B не входит в endpoints, A продолжает отвечать. <code>helm rollback</code> возвращает A; её версия проверена по фактическому ответу.</p>
<p>Отдельно выполнен SIGKILL PID 1 внутри собственной реплики: это авария, без preStop и graceful shutdown. Клиентский HTTP 200 с отсутствующим терминатором <code>done</code> учитывается как оборванный поток. Восстановление видно по новым Ready/Endpoints и последующим успешным запросам. После удаления реплики два диалога с разными CANARY продолжаются на другом Pod с той же A; запрос содержит всю историю. При SIGKILL процесс не может записать terminal log и увеличить Counter: аварийные обрывы считаются по клиентскому журналу; Counter каждой реплики может сброситься при рестарте, а rate/increase учитывают такой reset. Автоматических повторов нет (<code>retry=0</code>), скрытая успешная повторная попытка не маскирует обрыв.</p>
{image(1)}
<h2>3. Опыт B: ClientIP и горячая реплика</h2>
<p><strong>Гипотеза:</strong> IP-привязка при 80% трафика одного клиента переполняет очередь одной реплики, даже если у другой есть свободные слоты. Сравнение: обычный Service <code>sessionAffinity=None</code> и тот же selector с <code>ClientIP</code>, timeout=10800 с. Кеши перед каждым прогоном пусты; backend уже прогрет, лимиты и артефакт одинаковы. Два типа трафика × две политики × три повтора; каждая фаза 120 с, 12 запросов/с, 1440 запросов. Порядок политик чередуется между повторами.</p>
<p>В дополнительном контрпримере (если естественная IP-привязка не дала 2:2) явно подбирается отдельный Service с таким начальным соответствием клиентов репликам. Четыре клиентских Pod для дополнительного блока пересозданы на v2; их новые IP сохранены в <a href="results/counterexample-clients.json">counterexample-clients.json</a>. Исходные четыре IP основной серии сохранены в clients.json. Все пробные соответствия сохраняются в <a href="results/counterexample-selection.json">counterexample-selection.json</a>; это проверка возможности равномерного распределения, а не оценка вероятности такого события. Затем тот же равномерный сценарий повторяется трижды по 120 с. Четыре настоящих клиентских Pod: {esc(clients_text)}. kube-proxy и Services: <a href="results/network.json">network.json</a>, <a href="results/network-implementation.json">CNI / iptables и фактические affinity rules</a>. ClientIP использует исходный IP клиентского Pod до выбора backend; <code>peer</code> в журнале сервера подтверждает видимый приложению IP. Генератор открывает новое HTTP-соединение на каждую попытку (<code>force_close=True</code>), поэтому длительный keep-alive не создаёт дополнительную привязку в обычном Service. Пакетная реализация не обещает строгий round-robin, фактическое распределение показано в таблице.</p>
{table([n for n in summaries if n.startswith('B-')])}
{image(2)}{image(3)}
<p>Совместные снимки горячей очереди и свободных слотов второй реплики, а также Ready/EndpointSlice во время исходного дефекта: <a href="results/trace-proofs.json">trace-proofs.json</a>.</p><p><strong>Исправление:</strong> снять ClientIP, использовать обычный Service с соединением на запрос. Строки <code>B-api-skew-1..3</code> — повтор того же сценария после изменения политики. В этом эксперименте малое множество префиксов помещается в кеш каждой реплики, поэтому тёплая локальность восстанавливается на обеих. Выбор оценивается одновременно по TTFC, ошибкам и распределению; абсолютные числа относятся только к заданному имитатору.</p>
<p>Контрпример ожидаемого перекоса: {esc(', '.join(counter) if counter else 'равномерная привязка четырёх клиентов может случайно дать 2:2; конкретные записанные распределения приведены ниже. Если естественный прогон не дал такого случая, нужен отдельный зафиксированный опыт — он не объявляется полученным автоматически.')}.</p>
<details><summary>Все реплики: число запросов, задержка и очередь</summary>{pod_table()}</details>
<h2>4. Опыт C: средняя CPU, HPA и алерты</h2>
<p><strong>Гипотеза:</strong> вычислительная нагрузка поднимет CPU выше target и вызовет 2→3; после её снятия HPA вернёт 2 с задержкой стабилизации. Используется <code>autoscaling/v2</code>, min=2, max=3, target=55% от 200m, scaleUp window=0, scaleDown=60 с. Resource metrics API и CPU requests зафиксированы в <a href="results/checks/metrics-api.json">metrics API</a>. Условия HPA сохранены вместе с каждым snapshot. Создание и Ready новых Pod приведены ниже.</p>
{table([n for n in summaries if n.startswith('C-')])}
{image(4)}
<pre>{hpa_details()}</pre>
<p>Измеренное возвращение к минимуму: {esc([e for e in events if e['event']=='hpa_minimum_restored'])}. Опрос возврата к минимуму выполняется каждые 5 с, поэтому время обнаружения имеет погрешность до 5 с. Отсчёт ведётся от завершения нагрузки, поэтому сюда входят задержка метрик, цикл контроллера, окно стабилизации и удаление Pod. Настроенные 60 с не обещают ровно 60 с от последнего клиентского запроса.</p>
<p>Средняя CPU сравнивается с requests участвующих реплик. Для двух Pod с request=200m и target=55% средний порог — 110m/Pod; одна горячая реплика должна в среднем использовать больше 220m при почти нулевой CPU другой, чтобы поднять среднее выше порога (с учётом tolerance и алгоритма HPA). Очередь и sleeping chunks могут увеличиваться без пропорционального роста CPU. Новая реплика при ClientIP может остаться почти без запросов: существующие привязки сохраняются, HPA не переносит их на новый Pod. Повтор <code>C-hpa-hot-sticky</code> сохраняет это распределение в сырых ответах.</p>
<p>Алерты: TTFC p95 &gt;0,6 с и доля terminal outcomes, отличных от completed, &gt;10%. Для каждого нужен минимум 20 наблюдений за 30 с, удержание 8 с, evaluation_interval=2 с. Поэтому единичный медленный ответ не вызывает тревогу. Управляемая перегрузка длинной очереди даёт pending→firing, затем после снятия нагрузки и выхода из окна rate — норму. Наблюдавшиеся состояния: {esc(dict((k,sorted(v)) for k,v in states.items()))}. Полная последовательность: <a href="results/alerts.jsonl">alerts.jsonl</a>; определения: <a href="monitoring/alerts.yml">alerts.yml</a>.</p>
<h2>5. Методика измерений и Prometheus</h2>
<p>Монотонные часы на каждом клиенте: <code>time.perf_counter()</code>. TTFC = время первого содержательного JSON chunk − фактическая отправка; duration = конец чтения − отправка; inter-chunk latency = разности времён прихода соседних содержательных частей. Маркер done не является частью ответа. При N=1 массив интервалов пуст и среднее отсутствует. Метрики TTFT, TPOT и tokens/s не заявляются: фактических токенов LLM здесь нет. Для возможного реального backend: TTFT до первого содержательного токена, TPOT=(t_last−t_first)/(N−1), при одном токене TPOT=null; счётчик должен использовать tokenizer той же закреплённой модели.</p>
<p>В каждой строке сохранены schedule, sent, first, end, статус, outcome, Pod, фактическая версия, request ID, количество частей, интервалы и лаг отправки. Успех требует HTTP 200, done и ожидаемое количество частей. 429/503 — отклонённые, поток без done — interrupted, дедлайн клиента — timed_out. «Просроченные» здесь определены истечением timeout=35 с; лаг отправки считается отдельно и не включается в TTFC. Таблица содержит N, p50/p95 только успешных ответов, доли ошибок/тайм-аутов; завершённость не выводится из одного HTTP-статуса.</p>
<p>Гистограммы classic с границами 0,01; 0,025; 0,05; 0,1; 0,2; 0,4; 0,8; 1,5; 3; 5; 10; 20; 40; 80; +Inf с. PromQL для p95 по Pod:</p>
<pre>histogram_quantile(0.95,
  sum by (le, pod) (rate(inference_ttfc_seconds_bucket[30s])))
# Для всего стенда — sum by(le); усреднять p95 отдельных Pod нельзя.
sum by(pod)(rate(inference_requests_total{{outcome="completed"}}[30s]))</pre>
<p>Изолированная калибровка: один Kubernetes Pod, 80 запросов, три повторяющихся префикса, без постороннего трафика; разности накопительных bucket-счётчиков до/после задают тот же набор успешных ответов. Сравнение: {esc(hist)}. Реальный результат выполнения PromQL с временными @-границами сохранён в <a href="results/histogram-promql.json">histogram-promql.json</a>. Серверная TTFC исключает сеть и port-forward, а клиентская включает их; histogram_quantile линейно интерполирует внутри bucket, поэтому точного совпадения с эмпирическим p95 не ожидается.</p>
<p>Prometheus обнаруживает каждый Pod, а не балансируемый Service; Pod — ограниченная инфраструктурная метка. Исходы и hit/miss имеют фиксированный словарь. request/session ID находятся только в логах. Дашборд содержит TTFC, полную длительность, running/waiting, completed RPS, отказы, CPU/RAM и Ready; <a href="monitoring/dashboard.json">экспорт Grafana</a>. CPU/RAM получены из cAdvisor, Ready — из kube-state-metrics.</p>
<h2>6. Собственные проверки, ограничения и ревью</h2>
<p>Девять HTTP/контрактных тестов: прогрев/probes, изоляция диалогов, ограничение очереди с учётом отказов, оборванный поток/освобождение слота, повреждённый артефакт идентичность расписания с долей 80% воспроизведение исходного дефекта probes классификация HTTP 500 с текстовым телом и продолжение обслуживания старых сетевых соединений при снятой readiness во время preStop. Исходная запись аварийного bridge HTTP 500 сохранена со старым outcome=connection_error и фактическим status=500; это ошибка классификации клиента, устранённая регрессионным тестом. Она не влияет на долю неуспехов и не относится к штатному rollout. Kubernetes-проверки в run_lab дополнительно отправляют HTTP-запросы при прогреве, после отката, при SIGKILL и после замены Pod, проверяют версии и две контрольные строки. <a href="results/verification.json">Строгий gate сдачи</a> проверяет сырые доказательства, 12 полных сравнений, последовательности алертов и четыре графика; ненулевой exit означает незавершённый пункт.</p>
<p>Сохранён также первоначальный сбой Docker Desktop и ошибка разработки метаданных; они не выдаются за поведение исправленного приложения. Воспроизводимые итоговые измерения должны быть получены при доступном Docker и Kubernetes. Один узел kind не доказывает отказоустойчивость к потере узла, сеть другого CNI, производительность GPU, настоящий KV-cache, batching и порядок токенов. Клиентская история тестирует переносимость/изоляцию контекста, но не является проверкой качества ответов модели.</p>
<h3>Ревью PR: закрепление диалога</h3>
<pre class="prose">{esc((ROOT/'REVIEW.md').read_text())}</pre>
<h2>Приложения и запуск</h2>
<p><a href="README.md">README с командами</a> · <a href="chart/values.yaml">Helm values</a> · <a href="results/commands.jsonl">точные команды и время</a> · <a href="results/pod-logs/">логи Pod</a> · <a href="results/schedules/">расписания</a>. Репозиторий: <a href="https://github.com/redron4k/MLOps-course">redron4k/MLOps-course</a>; базовый коммит предыдущего ДЗ eca39d6. Коммит сдачи фиксируется после добавления файлов; README не подменяет его SHA старым значением.</p>
<p>Первичные технические источники: <a href="https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/">probes</a>, <a href="https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-termination-flow">Pod termination</a>, <a href="https://kubernetes.io/docs/concepts/services-networking/service/#session-affinity">Service affinity</a>, <a href="https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/">HPA</a>, <a href="https://prometheus.io/docs/practices/histograms/">histograms</a>, <a href="https://llm-d.ai/blog/sticky-until-saturated-token-aware-routing">Google / llm-d</a>.</p>'''
style='''*{box-sizing:border-box}body{margin:0;background:#f3f5fa;color:#182438;font:16px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1180px;margin:auto;padding:60px 44px 100px;background:white}h1{font-size:56px;line-height:1.08;letter-spacing:-2px;margin:18px 0 28px}h2{font-size:28px;margin-top:54px;line-height:1.3;border-top:1px solid #dae0ec;padding-top:26px}h3{font-size:22px}.lead{font-size:23px;color:#4d5870}.eyebrow{font-size:13px;letter-spacing:2px;color:#5948cf}a{color:#4f46b9}code{background:#f0f2f8;padding:2px 5px;border-radius:4px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f5fa;padding:22px;border-radius:8px;font-size:13px;line-height:1.6}.prose{font-family:inherit;font-size:15px}table{border-collapse:collapse;width:100%;font-size:12px;line-height:1.4;margin:24px 0}th{background:#edf0f8;text-align:left}td,th{padding:11px 8px;border-bottom:1px solid #e2e6ef;overflow-wrap:anywhere}svg{width:100%;height:auto;margin:22px 0}summary{cursor:pointer;color:#4f46b9}details{margin:30px 0}p{overflow-wrap:anywhere}@media(max-width:700px){main{padding:30px 18px}h1{font-size:40px}table{font-size:10px}td,th{padding:6px 3px}}@media print{body{background:white}main{padding:0}h2{break-after:avoid}svg{break-inside:avoid}table{font-size:9px}details{display:block}}'''
(ROOT/'report.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ДЗ 11 · Kubernetes под нагрузкой</title><style>'+style+'</style><main>'+body+'</main></html>')
print(f'Generated report.html, four SVG figures, {len(rows)} actual attempt records')
