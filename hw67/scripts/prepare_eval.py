"""Авторские MCQ и полный train-index; связь с тензорами ДЗ 4 проверяется повторным encode."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.contract import validate_golden

SOURCES = {
 'ui-recyclerview': 'https://developer.android.com/develop/ui/views/layout/recyclerview',
 'ui-dialog': 'https://developer.android.com/develop/ui/views/components/dialogs',
 'services': 'https://developer.android.com/develop/background-work/services',
 'firebase-auth': 'https://firebase.google.com/docs/auth/android/start',
 'firebase-messaging': 'https://firebase.google.com/docs/cloud-messaging/android/receive-messages',
}
# В каждой строке: вопрос, правильный вариант, два дистрактора.
QUESTIONS = {
'ui-recyclerview': [
 ('Кто размещает элементы RecyclerView?', 'LayoutManager', 'BroadcastReceiver', 'FirebaseAuth'),
 ('Кто связывает данные с ViewHolder?', 'Adapter', 'Service', 'Dialog'),
 ('Что хранит ViewHolder?', 'View элемента', 'Пароль пользователя', 'FCM token'),
 ('Зачем RecyclerView переиспользует View?', 'Чтобы сократить создание View', 'Чтобы удалить данные', 'Чтобы запросить разрешение'),
 ('Какой менеджер задаёт линейный список?', 'LinearLayoutManager', 'GridLayoutManager', 'FragmentManager'),
 ('Какой callback адаптера создаёт ViewHolder?', 'onCreateViewHolder', 'onBindViewHolder', 'getItemCount'),
 ('Какой callback записывает данные в готовый ViewHolder?', 'onBindViewHolder', 'onCreateViewHolder', 'onDestroy'),
 ('Что возвращает getItemCount()?', 'Число элементов', 'Высоту экрана', 'Номер потока'),
 ('Какой LayoutManager задаёт обычную сетку?', 'GridLayoutManager', 'LinearLayoutManager', 'FragmentManager'),
 ('Что сообщает адаптеру о вставке одного элемента?', 'notifyItemInserted', 'notifyItemRemoved', 'notifyItemChanged'),
],
'ui-dialog': [
 ('Какой компонент управляет жизненным циклом диалога?', 'DialogFragment', 'Service', 'RecyclerView'),
 ('В каком callback DialogFragment создают Dialog?', 'onCreateDialog', 'onBindViewHolder', 'onStartCommand'),
 ('Чем удобно построить AlertDialog?', 'AlertDialog.Builder', 'Intent.Builder', 'FirebaseAuth'),
 ('Какой диалог выбирает дату?', 'DatePickerDialog', 'TimePickerDialog', 'ProgressDialog'),
 ('Какой диалог выбирает время?', 'TimePickerDialog', 'DatePickerDialog', 'AlertDialog.Builder'),
 ('Какой метод Builder задаёт заголовок?', 'setTitle', 'setMessage', 'setView'),
 ('Какой метод Builder задаёт основное сообщение?', 'setMessage', 'setTitle', 'setItems'),
 ('Какой метод Builder добавляет кнопку подтверждения?', 'setPositiveButton', 'setNegativeButton', 'setNeutralButton'),
 ('Какой метод Builder добавляет собственный layout?', 'setView', 'setTitle', 'setCancelable'),
 ('Как задать список с несколькими выбранными пунктами?', 'setMultiChoiceItems', 'setSingleChoiceItems', 'setMessage'),
],
'services': [
 ('Создаёт ли обычный Android Service отдельный поток автоматически?', 'Нет', 'Да, всегда IO', 'Да, всегда новый Thread'),
 ('Есть ли у Service собственный пользовательский интерфейс?', 'Нет', 'Да, обязательная Activity', 'Да, обязательный Dialog'),
 ('В каком потоке по умолчанию вызываются callbacks Service?', 'Главном потоке процесса', 'Обязательно IO', 'Обязательно отдельном процессе'),
 ('Чем запущенный Service может остановить себя?', 'stopSelf()', 'finish()', 'dismiss()'),
 ('Как клиент связывается с bound Service?', 'bindService()', 'showDialog()', 'startActivity()'),
 ('Какой callback возвращает интерфейс для bound Service?', 'onBind()', 'onCreateDialog()', 'onBindViewHolder()'),
 ('Что возвращает onBind() клиенту?', 'IBinder', 'ViewHolder', 'AlertDialog'),
 ('Какой callback получает запрос запуска started Service?', 'onStartCommand()', 'onCreateDialog()', 'onMessageReceived()'),
 ('Какой callback вызывается при первом создании Service?', 'onCreate()', 'onBindViewHolder()', 'onActivityResult()'),
 ('Нужен ли пользовательский notification для foreground Service?', 'Да', 'Нет никогда', 'Только на эмуляторе'),
],
'firebase-auth': [
 ('Какой метод создаёт пользователя email/password?', 'createUserWithEmailAndPassword', 'signInWithEmailAndPassword', 'signOut'),
 ('Какой метод выполняет вход email/password?', 'signInWithEmailAndPassword', 'createUserWithEmailAndPassword', 'sendPasswordResetEmail'),
 ('Как получить текущего пользователя FirebaseAuth в Kotlin?', 'auth.currentUser', 'auth.currentActivity', 'auth.currentView'),
 ('Что обычно означает currentUser == null после инициализации Auth?', 'Пользователь не вошёл', 'Вход подтверждён', 'Пароль пустой'),
 ('Как выйти из FirebaseAuth?', 'signOut()', 'stopSelf()', 'dismiss()'),
 ('Какое поле FirebaseUser идентифицирует пользователя?', 'uid', 'layoutId', 'notificationId'),
 ('Как отправить письмо подтверждения email?', 'sendEmailVerification()', 'sendPasswordResetEmail()', 'signOut()'),
 ('Как инициировать сброс пароля по email?', 'sendPasswordResetEmail()', 'sendEmailVerification()', 'getIdToken()'),
 ('Как проверить успешность Task при входе?', 'task.isSuccessful', 'task.isVisible', 'task.isAttachedToWindow'),
 ('Где включить провайдера Email/Password?', 'Firebase console Authentication', 'RecyclerView.Adapter', 'Android Layout Editor'),
],
'firebase-messaging': [
 ('Какой Service обрабатывает FCM callbacks?', 'FirebaseMessagingService', 'FirebaseAuth', 'DialogFragment'),
 ('Какой callback принимает data-сообщение FCM?', 'onMessageReceived', 'onBindViewHolder', 'onCreateDialog'),
 ('Какой тип объекта приходит в onMessageReceived?', 'RemoteMessage', 'ViewHolder', 'FirebaseUser'),
 ('Как получить key-value data из RemoteMessage?', 'remoteMessage.data', 'remoteMessage.uid', 'remoteMessage.layout'),
 ('Куда вынести долгую работу из onMessageReceived?', 'В WorkManager', 'В бесконечный цикл callback', 'В onDraw'),
 ('Для legacy AndroidConfig v1: куда попадает notification-only в фоне?', 'В системную панель уведомлений', 'В RecyclerView', 'В FirebaseAuth'),
 ('Для legacy AndroidConfig v1: где data смешанного сообщения после тапа в фоне?', 'В extras Intent launcher Activity', 'В пароле FirebaseUser', 'В ViewHolder'),
 ('Как получить текущий FCM registration token?', 'FirebaseMessaging.getInstance().token', 'FirebaseAuth.getInstance().uid', 'AlertDialog.Builder'),
 ('Какой callback сообщает новый registration token?', 'onNewToken', 'onStartCommand', 'onBindViewHolder'),
 ('Можно ли считать registration token неизменным навсегда?', 'Нет, он может обновиться', 'Да, это вечный пароль', 'Да, это имя пакета'),
]}


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def main():
    import torch
    import yaml
    from transformers import AutoTokenizer
    hw4 = ROOT.parent / 'hw4'
    raw = [json.loads(l) for l in (hw4 / 'data/train.jsonl').read_text().splitlines() if l.strip()]
    # Тот же ключ группировки, что в hw3/src/split.py.
    sys.path.insert(0, str(ROOT.parent / 'hw3'))
    from src.textnorm import normalize_group
    training = [dict(id=r['id'], input=r['messages'][1]['content'], group_id=normalize_group(r['topic'])) for r in raw]
    if set(QUESTIONS) & {r['group_id'] for r in training}:
        raise ValueError('Тема golden входит в train')
    rows = []
    labels = ['A', 'B', 'C']
    for topic, questions in QUESTIONS.items():
        for i, (question, right, wrong1, wrong2) in enumerate(questions):
            options = [right, wrong1, wrong2]
            shift = i % 3
            options = options[shift:] + options[:shift]
            source = SOURCES[topic]
            if topic == 'firebase-auth' and 4 <= i <= 7:
                source = 'https://firebase.google.com/docs/auth/android/manage-users'
            if topic == 'firebase-messaging' and i >= 7:
                source = 'https://firebase.google.com/docs/cloud-messaging/android/get-started'
            if topic == 'ui-recyclerview' and i >= 5:
                source = 'https://developer.android.com/reference/androidx/recyclerview/widget/RecyclerView.Adapter'
            if topic == 'ui-dialog' and i >= 5:
                source = 'https://developer.android.com/reference/android/app/AlertDialog.Builder'
            rows.append(dict(id=f'{topic}-{i+1:02}', group_id=topic, input=question + '\n' + '\n'.join(f'{l}: {o}' for l,o in zip(labels,options)),
                             choices=labels.copy(), expected=labels[options.index(right)], slices=[topic], source=source,
                             review='Составлено и сверено агентом с документацией; человеческий просмотр перед сдачей обязателен.'))
    format_case = dict(rows[0], id='behavior-format', type='format')
    paraphrase = dict(rows[0], id='behavior-paraphrase', type='paraphrase',
                      input='Кто отвечает за расположение дочерних View внутри RecyclerView?\nA: Service\nB: LayoutManager\nC: FirebaseAuth', expected='B')
    refusal = dict(id='behavior-refusal', group_id='out-of-domain', type='refusal',
                   input='Назови столицу Франции.\nA: Париж\nB: Рим\nREFUSE: запрос вне Android-разработки',
                   choices=['A', 'B', 'REFUSE'], expected='REFUSE', slices=['out-of-domain'], source='Граница темы в evaluation/params.json')
    behavior = [format_case, paraphrase, refusal]
    proof = validate_golden(rows, training, behavior)
    dst = ROOT / 'data/eval'
    jsonl(dst / 'golden.jsonl', rows); jsonl(dst / 'training-index.jsonl', training); jsonl(dst / 'behavioral.jsonl', behavior)
    # Сравнение всех тензоров с повторным encode тем же кодом ДЗ 4.
    for key in list(sys.modules):
        if key == 'src' or key.startswith('src.'):
            del sys.modules[key]
    sys.path.insert(0, str(hw4))
    from src.tokenize_data import encode_example
    params = yaml.safe_load((hw4 / 'params.yaml').read_text())
    revision = yaml.safe_load((ROOT / 'params.yaml').read_text())['model']['revision']
    tok = AutoTokenizer.from_pretrained(params['model']['name'], revision=revision, local_files_only=True)
    blob = torch.load(ROOT / 'data/tokenized/train.pt', weights_only=True)
    encoded, dropped = [], []
    for record in raw:
        value = encode_example(tok, record, params, params['tokenize']['max_seq_len'])
        meta = value.pop('_meta')
        if meta['supervised']:
            encoded.append(value)
        else:
            dropped.append(record['id'])
    if encoded != blob['examples']:
        raise ValueError('Тензоры не воспроизводятся из полного train-index и HW4 кода')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    proof.update(raw_train_sha256=sha(hw4 / 'data/train.jsonl'), train_tensor_sha256=sha(ROOT / 'data/tokenized/train.pt'),
                 hw4_train_tensor_sha256=sha(hw4 / 'data/tokenized/train.pt'),
                 val_tensor_sha256=sha(ROOT / 'data/tokenized/val.pt'), hw4_params_sha256=sha(hw4 / 'params.yaml'),
                 hw4_tokenizer_code={p.name: sha(p) for p in sorted((hw4 / 'src').glob('*.py'))},
                 tensor_rows=len(encoded), dropped_no_supervision=dropped, all_encoded_examples_equal=True,
                 model_revision=revision)
    print(json.dumps(proof, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
