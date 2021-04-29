function
main(splash, args)

splash.private_mode_enabled = false

--[[splash: set_user_agent(
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15")
headers = {[
    "User-Agent"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15"}
splash: set_custom_headers(headers)]]
splash: on_request(function(request)
request: set_header('User-Agent',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15')
end)
assert (splash:go(args.url))
assert (splash:wait(0.5))

input_box =
assert (splash:select("#_esgratingsprofile_keywords"))
input_box: focus()
input_box: send_text("michelin")
assert (splash:wait(5))

button =
assert (splash:select("#ui-id-1>li"))
assert (splash:wait(3))
button: mouse_click()
--[[input_box: send_keys("<Enter>")]]
assert (splash:wait(3))
splash: set_viewport_full()

return {
    html = splash: html(),
png = splash:png(),
}
end