"""Generate fixed-version small Prometheus/Grafana/kube-state-metrics manifests."""
from pathlib import Path
import json
import yaml
ns='hw11'
objects=[]
def add(kind,name,**kw):
    obj=dict(apiVersion='v1',kind=kind,metadata={'name':name,'namespace':ns},**kw)
    objects.append(obj);return obj
add('ServiceAccount','monitor')
objects.append(dict(apiVersion='rbac.authorization.k8s.io/v1',kind='ClusterRole',metadata={'name':'hw11-monitor'},rules=[
    {'apiGroups':[''],'resources':['pods','nodes','nodes/proxy','services','endpoints','namespaces','replicationcontrollers','resourcequotas','limitranges','persistentvolumes','persistentvolumeclaims'],'verbs':['get','list','watch']},
    {'apiGroups':['apps'],'resources':['deployments','replicasets','statefulsets','daemonsets'],'verbs':['get','list','watch']},
    {'apiGroups':['batch'],'resources':['jobs','cronjobs'],'verbs':['get','list','watch']},
    {'apiGroups':['autoscaling'],'resources':['horizontalpodautoscalers'],'verbs':['get','list','watch']},
    {'nonResourceURLs':['/metrics','/metrics/cadvisor'],'verbs':['get']}]))
objects.append(dict(apiVersion='rbac.authorization.k8s.io/v1',kind='ClusterRoleBinding',metadata={'name':'hw11-monitor'},roleRef={'apiGroup':'rbac.authorization.k8s.io','kind':'ClusterRole','name':'hw11-monitor'},subjects=[{'kind':'ServiceAccount','name':'monitor','namespace':ns}]))
add('ConfigMap','prom-config',data={p.name:p.read_text() for p in map(Path,['monitoring/prometheus.yml','monitoring/alerts.yml'])})
add('ConfigMap','grafana-provision',data={'datasource.yaml':yaml.safe_dump({'apiVersion':1,'datasources':[{'name':'Prometheus','uid':'prometheus','type':'prometheus','url':'http://prometheus:9090','isDefault':True}]}),'dashboard.yaml':yaml.safe_dump({'apiVersion':1,'providers':[{'name':'hw11','type':'file','options':{'path':'/dashboards'}}]})})
add('ConfigMap','grafana-dashboard',data={'hw11.json':Path('monitoring/dashboard.json').read_text()})

def deployment(name,image,port,args=None,volumes=None,mounts=None,env=None,resources=None):
    container={'name':name,'image':image,'ports':[{'containerPort':port}],'resources':resources or {'requests':{'cpu':'50m','memory':'96Mi'},'limits':{'cpu':'500m','memory':'384Mi'}}}
    if args:container['args']=args
    if mounts:container['volumeMounts']=mounts
    if env:container['env']=[{'name':k,'value':v} for k,v in env.items()]
    spec={'serviceAccountName':'monitor','containers':[container]}
    if volumes:spec['volumes']=volumes
    objects.append({'apiVersion':'apps/v1','kind':'Deployment','metadata':{'name':name,'namespace':ns},'spec':{'replicas':1,'selector':{'matchLabels':{'app':name}},'template':{'metadata':{'labels':{'app':name}},'spec':spec}}})
    add('Service',name,spec={'selector':{'app':name},'ports':[{'port':port,'targetPort':port}]})

deployment('prometheus','prom/prometheus:v3.2.1',9090,
    ['--config.file=/etc/prometheus/prometheus.yml','--storage.tsdb.path=/prometheus','--storage.tsdb.retention.time=2h','--storage.tsdb.retention.size=256MB'],
    [{'name':'config','configMap':{'name':'prom-config'}},{'name':'data','emptyDir':{'sizeLimit':'512Mi'}}],
    [{'name':'config','mountPath':'/etc/prometheus'},{'name':'data','mountPath':'/prometheus'}])
deployment('grafana','grafana/grafana:11.6.0',3000,None,
    [{'name':'provision','configMap':{'name':'grafana-provision'}},{'name':'dashboards','configMap':{'name':'grafana-dashboard'}},{'name':'data','emptyDir':{'sizeLimit':'128Mi'}}],
    [{'name':'provision','mountPath':'/etc/grafana/provisioning/datasources/datasource.yaml','subPath':'datasource.yaml'}, {'name':'provision','mountPath':'/etc/grafana/provisioning/dashboards/dashboard.yaml','subPath':'dashboard.yaml'}, {'name':'dashboards','mountPath':'/dashboards'}, {'name':'data','mountPath':'/var/lib/grafana'}],
    {'GF_AUTH_ANONYMOUS_ENABLED':'true','GF_AUTH_ANONYMOUS_ORG_ROLE':'Viewer','GF_ANALYTICS_REPORTING_ENABLED':'false','GF_ANALYTICS_CHECK_FOR_UPDATES':'false'})
deployment('kube-state-metrics','registry.k8s.io/kube-state-metrics/kube-state-metrics:v2.15.0',8080,['--resources=pods,deployments,horizontalpodautoscalers'])
Path('monitoring/stack.yaml').write_text(yaml.safe_dump_all(objects,sort_keys=False))
