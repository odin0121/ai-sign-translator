document.addEventListener('DOMContentLoaded', () => {
    const modelProfileSelect = document.getElementById('model-profile-select');
    const modelProfileStatus = document.getElementById('model-profile-status');
    const learningSignList = document.getElementById('learning-sign-list');
    const learningSignCount = document.getElementById('learning-sign-count');
    const learningSearch = document.getElementById('learning-search');
    const learningProgressLabel = document.getElementById('learning-progress-label');
    const learningProgressBar = document.getElementById('learning-progress-bar');
    const learningSignCategory = document.getElementById('learning-sign-category');
    const learningSignTitle = document.getElementById('learning-sign-title');
    const learningSignDescription = document.getElementById('learning-sign-description');
    const learningDemoVideo = document.getElementById('learning-demo-video');
    const learningDemoImage = document.getElementById('learning-demo-gif');
    const learningDemoPlaceholder = document.getElementById('learning-demo-placeholder');
    const demoSourceBadge = document.getElementById('demo-source-badge');
    const learningFeedback = document.getElementById('learning-feedback');
    const practiceSignLink = document.getElementById('practice-sign-btn');
    const markLearnedButton = document.getElementById('mark-learned-btn');
    const nextLessonButton = document.getElementById('next-lesson-btn');
    const lessonLearnedBadge = document.getElementById('lesson-learned-badge');
    const themeToggle = document.getElementById('theme-toggle-btn');
    const learningModelOptions = document.querySelectorAll('.learning-model-option');

    let selectedModelProfile = modelProfileSelect?.value || 'wlasl100';
    const requestedProfile = new URLSearchParams(window.location.search).get('model_profile');
    if (modelProfileSelect && requestedProfile) {
        const validProfile = [...modelProfileSelect.options].some(option => option.value === requestedProfile);
        if (validProfile) {
            modelProfileSelect.value = requestedProfile;
            selectedModelProfile = requestedProfile;
        }
    }

    let lessons = [];
    let selectedSign = null;
    let learnedSigns = new Set();
    let profileRequestId = 0;

    function applyTheme(theme) {
        const light = theme === 'light';
        document.body.classList.toggle('light-theme', light);
        document.body.classList.toggle('dark-theme', !light);
        if (themeToggle) {
            themeToggle.querySelector('.icon-sun').style.display = light ? 'none' : 'inline-block';
            themeToggle.querySelector('.icon-moon').style.display = light ? 'inline-block' : 'none';
            themeToggle.querySelector('.theme-label').textContent = light ? 'Dark' : 'Bright';
        }
        localStorage.setItem('app-theme', theme);
    }

    applyTheme(localStorage.getItem('app-theme') || 'dark');
    themeToggle?.addEventListener('click', () => {
        applyTheme(document.body.classList.contains('light-theme') ? 'dark' : 'light');
    });

    function setFeedback(message, state = '') {
        learningFeedback.textContent = message;
        learningFeedback.classList.toggle('success', state === 'success');
        learningFeedback.classList.toggle('try-again', state === 'try-again');
    }

    function updateLearningModelOptions() {
        learningModelOptions.forEach(option => {
            const selected = option.dataset.modelProfile === selectedModelProfile;
            option.classList.toggle('selected', selected);
            option.setAttribute('aria-pressed', String(selected));
        });
    }

    function updateProgress() {
        const learnedCount = lessons.filter(lesson => learnedSigns.has(lesson.token)).length;
        const total = lessons.length;
        learningProgressLabel.textContent = `${learnedCount} / ${total} learned`;
        learningProgressBar.style.width = `${total ? learnedCount / total * 100 : 0}%`;
    }

    function renderLessonList() {
        const query = learningSearch.value.trim().toLowerCase();
        const filtered = lessons.filter(lesson =>
            lesson.token.toLowerCase().includes(query)
            || (lesson.category || '').toLowerCase().includes(query)
        );
        learningSignList.replaceChildren();

        if (!filtered.length) {
            const empty = document.createElement('div');
            empty.className = 'learning-list-empty';
            empty.textContent = lessons.length ? 'No signs match your search.' : 'No lessons are available for this model.';
            learningSignList.appendChild(empty);
            return;
        }

        filtered.forEach(lesson => {
            const button = document.createElement('button');
            button.className = 'learning-sign-option';
            button.type = 'button';
            button.dataset.token = lesson.token;
            button.setAttribute('aria-pressed', String(lesson.token === selectedSign));
            if (lesson.token === selectedSign) button.classList.add('selected');

            const index = document.createElement('span');
            index.className = 'lesson-index';
            index.textContent = String(lessons.indexOf(lesson) + 1).padStart(2, '0');
            const token = document.createElement('span');
            token.className = 'lesson-token';
            token.textContent = lesson.token;
            button.append(index, token);

            if (learnedSigns.has(lesson.token)) {
                const learnedMark = document.createElement('span');
                learnedMark.className = 'mini-learned';
                learnedMark.setAttribute('aria-label', 'Learned');
                button.appendChild(learnedMark);
            }
            learningSignList.appendChild(button);
        });
    }

    function showDemoError() {
        if (learningDemoVideo) learningDemoVideo.hidden = true;
        learningDemoImage.hidden = true;
        learningDemoPlaceholder.hidden = false;
        learningDemoPlaceholder.querySelector('strong').textContent = 'Could not load this example';
        learningDemoPlaceholder.querySelector('span:last-child').textContent =
            'Try another sign or check that the source dataset is available.';
    }

    function selectSign(token) {
        const lesson = lessons.find(item => item.token === token);
        if (!lesson) return;
        selectedSign = token;
        renderLessonList();

        learningSignCategory.textContent = lesson.category || 'SIGN LESSON';
        learningSignTitle.textContent = lesson.token;
        learningSignDescription.textContent = lesson.description || 'Watch the recorded training example, then practice the sign.';
        lessonLearnedBadge.hidden = !learnedSigns.has(token);
        markLearnedButton.disabled = false;
        markLearnedButton.textContent = learnedSigns.has(token) ? 'Mark not learned' : 'Mark as learned';
        nextLessonButton.disabled = lessons.length < 2;

        const practiceUrl = new URL('/', window.location.origin);
        practiceUrl.searchParams.set('model_profile', selectedModelProfile);
        practiceUrl.searchParams.set('practice_sign', lesson.token);
        practiceSignLink.href = practiceUrl.toString();
        practiceSignLink.removeAttribute('aria-disabled');
        practiceSignLink.classList.remove('disabled');

        if (learningDemoVideo) {
            learningDemoVideo.pause();
            learningDemoVideo.removeAttribute('src');
            learningDemoVideo.hidden = true;
            learningDemoVideo.load();
            learningDemoVideo.onerror = null;
        }
        learningDemoImage.removeAttribute('src');
        learningDemoImage.hidden = true;
        learningDemoImage.onerror = null;
        learningDemoImage.alt = `Recorded training example of ${lesson.token}`;
        demoSourceBadge.hidden = !lesson.demo_url;
        learningDemoPlaceholder.hidden = Boolean(lesson.demo_url);
        learningDemoPlaceholder.querySelector('strong').textContent =
            lesson.demo_url ? 'Loading sign example…' : 'Example unavailable';
        learningDemoPlaceholder.querySelector('span:last-child').textContent = lesson.demo_url
            ? 'Preparing the recorded training example.'
            : 'The source dataset is not available on this server.';

        if (lesson.demo_url && lesson.demo_format === 'video/mp4' && learningDemoVideo) {
            learningDemoVideo.onerror = showDemoError;
            learningDemoVideo.src = lesson.demo_url;
            learningDemoVideo.hidden = false;
            learningDemoVideo.load();
            learningDemoVideo.play().catch(() => {});
        } else if (lesson.demo_url && lesson.demo_format?.startsWith('image/')) {
            learningDemoImage.onload = () => { learningDemoPlaceholder.hidden = true; };
            learningDemoImage.onerror = showDemoError;
            learningDemoImage.src = lesson.demo_url;
            learningDemoImage.hidden = false;
        }
        setFeedback('Examples are recordings from training data; signing can vary by person and region.');
    }

    async function loadModelProfile(modelProfile) {
        const requestId = ++profileRequestId;
        selectedModelProfile = modelProfile;
        selectedSign = null;
        learningSignList.innerHTML = '<div class="learning-list-empty">Loading sign lessons…</div>';
        try {
            const response = await fetch(`/api/vocabulary/?model_profile=${encodeURIComponent(modelProfile)}`);
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
            if (requestId !== profileRequestId) return;

            lessons = data.vocabulary;
            learningSignCount.textContent = `${data.count} signs`;
            modelProfileStatus.textContent = data.model_ready
                ? `${data.test_accuracy_percent}% held-out accuracy`
                : 'Model not trained';
            try {
                const saved = JSON.parse(localStorage.getItem(`signa-learned-${modelProfile}`) || '[]');
                learnedSigns = new Set(Array.isArray(saved) ? saved : []);
            } catch (error) {
                console.warn('Could not load learning progress:', error);
                learnedSigns = new Set();
            }

            learningSearch.value = '';
            updateProgress();
            renderLessonList();
            if (lessons.length) selectSign(lessons[0].token);
        } catch (error) {
            if (requestId !== profileRequestId) return;
            console.error('Could not load learning vocabulary:', error);
            learningSignCount.textContent = 'Unavailable';
            learningSignList.innerHTML = '<div class="learning-list-empty">Could not load lessons. Please retry.</div>';
            modelProfileStatus.textContent = 'Model unavailable';
        }
    }

    modelProfileSelect?.addEventListener('change', () => {
        selectedModelProfile = modelProfileSelect.value;
        updateLearningModelOptions();
        const url = new URL(window.location.href);
        url.searchParams.set('model_profile', modelProfileSelect.value);
        window.history.replaceState({}, '', url);
        loadModelProfile(modelProfileSelect.value);
    });
    learningModelOptions.forEach(option => {
        option.addEventListener('click', () => {
            if (!modelProfileSelect || !option.dataset.modelProfile) return;
            modelProfileSelect.value = option.dataset.modelProfile;
            modelProfileSelect.dispatchEvent(new Event('change'));
        });
    });
    learningSearch.addEventListener('input', renderLessonList);
    learningSignList.addEventListener('click', event => {
        const button = event.target.closest('.learning-sign-option');
        if (button) selectSign(button.dataset.token);
    });
    markLearnedButton.addEventListener('click', () => {
        if (!selectedSign) return;
        if (learnedSigns.has(selectedSign)) learnedSigns.delete(selectedSign);
        else learnedSigns.add(selectedSign);
        try {
            localStorage.setItem(`signa-learned-${selectedModelProfile}`, JSON.stringify([...learnedSigns]));
        } catch (error) {
            console.warn('Could not save learning progress:', error);
            setFeedback('Progress changed for this session, but could not be saved in this browser.');
        }
        updateProgress();
        renderLessonList();
        lessonLearnedBadge.hidden = !learnedSigns.has(selectedSign);
        markLearnedButton.textContent = learnedSigns.has(selectedSign) ? 'Mark not learned' : 'Mark as learned';
    });
    nextLessonButton.addEventListener('click', () => {
        if (lessons.length < 2 || !selectedSign) return;
        const index = lessons.findIndex(lesson => lesson.token === selectedSign);
        selectSign(lessons[(index + 1) % lessons.length].token);
    });

    updateLearningModelOptions();
    loadModelProfile(selectedModelProfile);
});