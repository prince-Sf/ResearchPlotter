from research_plotter.ui.workbench_shell import render_theme_styles


class _StyleCapture:
    def __init__(self, dark: bool):
        self.session_state = {"rp_theme_dark": dark}
        self.styles: list[str] = []

    def markdown(self, value: str, **_kwargs) -> None:
        self.styles.append(value)


def test_shared_theme_tokens_and_button_states_are_defined_for_both_themes():
    light = _StyleCapture(dark=False)
    dark = _StyleCapture(dark=True)

    render_theme_styles(light)
    render_theme_styles(dark)

    light_css = "\n".join(light.styles)
    dark_css = "\n".join(dark.styles)

    # The token names consumed by the component rules must match the names
    # emitted from Python palette keys (underscores are normalized to dashes).
    assert "--rp-button-secondary-bg: #FFFFFF" in light_css
    assert "--rp-button-secondary-text: #17202C" in light_css
    assert "--rp-button-primary-bg: #176C45" in light_css
    assert "--rp-button-primary-text: #FFFFFF" in light_css
    assert "--rp-button-secondary-bg: #131720" in dark_css
    assert "--rp-button-secondary-text: #F2F6FA" in dark_css
    assert "--rp-button-secondary-border: #203443" in dark_css
    assert "--rp-button-secondary-hover-text: #39E6BB" in dark_css
    assert "--rp-button-primary-text: #08111B" in dark_css
    assert "--rp-success-bg: #E5F4E8" in light_css
    assert "--rp-success-text: #14532D" in light_css
    assert "--rp-warning-text: #744B00" in light_css
    assert "--rp-error-text: #8B1E1A" in light_css
    assert "--rp-info-text: #163D63" in light_css
    assert "--rp-success-bg: #143323" in dark_css
    assert "--rp-success-text: #B7F7CC" in dark_css

    for css in (light_css, dark_css):
        assert "button[data-testid=\"stBaseButton-secondary\"]" in css
        assert "button:focus-visible" in css
        assert "button:disabled" in css
        assert "padding-top:var(--rp-native-header-height)" in css
        assert "--rp-native-header-height:3.75rem" in css
        assert "input::placeholder" in css
        assert "rp-top-status" in css
        assert '[data-testid="stWidgetLabel"]' in css
        assert '[data-testid="stFileUploaderDropzoneInstructions"]' in css
        assert '[data-testid="stAlertContentSuccess"]' in css
        assert '[data-testid="stAlertContentWarning"]' in css
        assert '[data-testid="stAlertContentError"]' in css
        assert '[data-testid="stAlertContentInfo"]' in css
        assert 'section[data-testid="stSidebar"][aria-expanded="false"]' in css
        assert 'button[data-testid="stExpandSidebarButton"]' in css
        assert '[data-testid="stSidebarCollapseButton"] button' in css
        assert 'visibility:visible !important' in css
        assert '[data-testid="stSidebarContent"] > [data-testid="stSidebarHeader"]' in css
        assert '> [data-testid="stColumn"]:first-child' in css
        assert '> [data-testid="stColumn"]:nth-child(2)' in css
        assert '> [data-testid="stColumn"]:nth-child(3)' in css
        assert 'grid-template-columns:repeat(2,minmax(0,1fr))' in css
        assert ':has(> [data-testid="stColumn"] .rp-metric-icon)' in css
        assert 'input[role="switch"]' in css
        assert 'input:focus-visible' in css
        assert '[data-testid="stProgress"]' in css


def test_theme_token_names_do_not_leave_button_tokens_in_underscore_form():
    styles = _StyleCapture(dark=False)
    render_theme_styles(styles)
    css = "\n".join(styles.styles)

    assert "--rp-button_secondary_bg" not in css
    assert "var(--rp-button-secondary-bg)" in css
