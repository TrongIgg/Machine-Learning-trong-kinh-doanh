document.addEventListener('DOMContentLoaded', function () {
    var form = document.getElementById('survey-form');
    if (!form) return;

    var steps = [
        document.getElementById('demand-section'),  // Thay bằng ID thực của section 1 nếu có
        document.getElementById('specs-section'),  // Section 2
        document.getElementById('profile-section') // Section 3
    ];
    var dividers = [
        document.getElementById('specs-divider'),
        document.getElementById('profile-divider')
    ];
    var consentGroup = document.getElementById('i8uzcc');
    var privacyNotice = document.getElementById('iol8m9');

    var prevBtn = form.querySelector('.msf-prev');
    var nextBtn = form.querySelector('.msf-next');
    var submitBtn = form.querySelector('.msf-submit');

    var stepText = document.getElementById('msf-step-text');
    var barFill = document.getElementById('msf-bar-fill');
    var stepItems = Array.from(form.querySelectorAll('.msf-step'));

    var current = 0;
    var total = steps.length;

    function showStep(index) {
        current = Math.max(0, Math.min(index, total - 1));
        steps.forEach((el, i) => el.style.display = i === current ? '' : 'none');
        if (dividers[0]) dividers[0].style.display = current === 0 ? '' : 'none';
        if (dividers[1]) dividers[1].style.display = current === 1 ? '' : 'none';
        if (consentGroup) consentGroup.style.display = current === total - 1 ? '' : 'none';
        if (privacyNotice) privacyNotice.style.display = current === total - 1 ? '' : 'none';
        if (prevBtn) prevBtn.classList.toggle('is-hidden', current === 0);
        if (nextBtn) nextBtn.classList.toggle('is-hidden', current === total - 1);
        if (submitBtn) submitBtn.classList.toggle('is-hidden', current !== total - 1);
        if (stepText) stepText.textContent = 'Step ' + (current + 1) + ' of ' + total;
        if (barFill) barFill.style.width = ((current + 1) / total * 100).toFixed(2) + '%';
        stepItems.forEach((item, i) => item.classList.toggle('is-active', i === current));
    }

    function goNext() { showStep(current + 1); }
    function goPrev() { showStep(current - 1); }

    if (nextBtn) nextBtn.addEventListener('click', goNext);
    if (prevBtn) prevBtn.addEventListener('click', goPrev);

    form.addEventListener('submit', (e) => {
        if (current < total - 1) {
            e.preventDefault();
            goNext();
        }
    });

    form.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && current < total - 1) {
            e.preventDefault();
            goNext();
        }
    });

    stepItems.forEach((item, i) => {
        item.style.cursor = 'pointer';
        item.addEventListener('click', () => showStep(i));
    });

    showStep(0);  // Khởi tạo
});