function AnimationRunner() {
	var queue = [];
	function step(timestamp) {
		if (!paused) {
			var now = Date.now();
			queue.filter(anim => anim[0] < now).sort((a,b) => a[0]-b[0]).forEach(anim => anim[1]())
			queue = queue.filter(anim => anim[0] >= now)
		}
		window.requestAnimationFrame(step);
	}
	
	window.requestAnimationFrame(step);
	
	var paused = false;
	var pauseTime = null
	window.onkeydown = function () {
		paused = !paused;
		if (paused)
			pauseTime = Date.now();
		else 
			queue = queue.map(anim => [anim[0] - pauseTime + Date.now(), anim[1]])
	}
	
	return function (fn, time) {
		queue.push([Date.now() + time, fn]);
	}
}

var run = AnimationRunner();

function words_to_span(words) {
	return words.map(word => {
		var span = document.createElement('span')
		if (word != '<null>')
			span.appendChild(document.createTextNode(word + ' '));
		return span;
	})
}

function assign_opacity(span, opacity) {
	span.style.opacity = opacity;
	//span.style.color = 'hsl(0, ' + Math.max(0, opacity*100 - 50)+ '%, 50%)';
}

function load_item(item) {
	return new Promise(function (resolve, reject) {
		var input_container = document.querySelector('#article');
		var summary_container = document.querySelector('#summary');
		input_container.innerHTML = '';
		summary_container.innerHTML = '';
		
		var input_spans = words_to_span(item['input']);
		input_spans.forEach(span => {
			span.style.opacity = 0.0
			input_container.appendChild(span)
		});
		run(() => { input_spans.forEach(span => { span.style.opacity = 1.0 }) }, 10);
		
		function set_attention(k) {
			console.log('set_attention', k)
			/*
			item['attention'][k].forEach((p, i) => {
				input_spans[i].style.opacity = 0.1 + 0.9*Math.sqrt(p)/Q
			})*/
			input_spans.forEach(span => { span.style.opacity = 0.1; })
			
			var attention = item['input'].map((input, i) => {
				var slice = item['attention'][k].slice(Math.max(i - 2, 0), Math.min(i + 3, item['input'].length -1));
				return slice.reduce((a,x) => a + x, 0) / slice.length || 0;
			});
			var max_p = Math.max.apply(null, attention)
			input_spans.forEach((span, i) => assign_opacity(span, 0.1 + 0.9*(attention[i] / max_p)))
		}
		
		function set_last_summary_word(k) {
			console.log('set_last_summary_word')
			if (summary_container.lastChild) {
				summary_container.lastChild.innerHTML = '';
				var span = words_to_span(item['choices'][k])[0]
				summary_container.lastChild.appendChild(span)
				span.style.opacity = 1.0
			}
		}
		
		function add_summary_words(k) {
			console.log('add_summary_word', k)
			var div = document.createElement('div');
			var choices_spans = words_to_span(item['choices'][k]);
			choices_spans.forEach((span, j) => {
				div.appendChild(span);
				assign_opacity(span, 0.1 + 0.9*(item['probs'][k][j]/item['probs'][k][0]));
			});
			div.style.opacity = 0.0;
			summary_container.appendChild(div)
			
			run(() => {
				div.style.opacity = 1.0;
			}, 20);
			
			// Fade out other choices
			run(() => {
				choices_spans.forEach((span, j) => { j != 0 && assign_opacity(span, 0.0); })
			}, 1700);
			
		}
		
		function next(k) {
			var t = 0;
			run(() => set_attention(k), t)
			run(() => add_summary_words(k), t+=100)
			run(() => set_last_summary_word(k), t+=2000)
			
			// Done
			if (item['choices'][k][0] == '<e>' || k+1 >= item['choices'].length) {
				
				run(() => {
					input_spans.forEach(span => { span.style.opacity = 0.0; });
					[].slice.call(summary_container.children).forEach(div => { div.style.opacity = 0.0; })
				}, t+=1800)
				run(resolve, t+=900);
			}
			
			// Next word
			else
				run(() => next(k+1), t+=50)
		}
		run(() => next(0), 2000);
	});
}

function visualize(data) {
	data = data.sort(() => Math.random() - 0.5)
	var i = 0;
	function next() {
		load_item(data[i++ % data.length]).then(next)
	}
	next()
}