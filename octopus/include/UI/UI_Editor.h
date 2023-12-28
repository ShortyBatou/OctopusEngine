#pragma once
#include <map>
#include <Core/Entity.h>
#include <Manager/DebugManager.h>
#include "UI/UI_Primitive.h"
class UI_Editor : public UI_Display {
public:
    UI_Editor() : _step(true) { }

    char* name() {
        return "Editor";
    }

    void init() override {
        for (UI_Display* ui : _core_ui) ui->init();
        for (UI_Component* ui : _managers_ui) ui->init();
        for (UI_Component* ui : _components_ui) ui->init();
        _step = true;
        _nb_step = 1;
        _count_step = 0;
    }

    void draw() override {
        if (_step) {
            _count_step++;
            if (_count_step >= _nb_step) {
                stop();
                _step = false;
                _count_step = 0;
            }
        }
        
        unsigned int wx, wy;
        AppInfo::Window_sizes(wx, wy);

        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(10, 10));
        ImGui::SetNextWindowSize(ImVec2(500, wy - 20));
        ImGui::Begin(name());
        if (ImGui::Button("Pause")) {
            _step = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Step")) {
            _step = true;
            _count_step = 0;
            play();
        }

        ImGui::SameLine();
        if (ImGui::Button("Resume")) {
            _step = false;
            play();
        }

        ImGui::InputInt("nb steps", &_nb_step);
        

        for (UI_Display* ui : _core_ui) {
            ui->draw();
        }
        
        if (ImGui::CollapsingHeader("Global", ImGuiTreeNodeFlags_DefaultOpen))
        {
            Entity* root = Engine::GetEntity(0);
            for (UI_Component* ui : _managers_ui) {
                if (!ui->can_draw(root)) continue;

                if (ImGui::TreeNodeEx(ui->name(), ImGuiTreeNodeFlags_DefaultOpen)) {
                    ui->draw(root);
                    ImGui::TreePop();
                    ImGui::Spacing();
                }
            }
        }

        if (ImGui::CollapsingHeader("Entities"))
        {
            for (Entity* entity : Engine::GetEntities()) {
                if (entity->id() == 0) continue;
                if (!ImGui::TreeNodeEx(entity->name().c_str())) continue;

                for (UI_Component* ui : _components_ui) {
                    if (!ui->can_draw(entity)) continue;
                    if (!ImGui::TreeNodeEx(ui->name())) continue;
                    ui->draw(entity);
                    ImGui::TreePop();
                    ImGui::Spacing();
                }
                ImGui::TreePop();
                ImGui::Spacing();
            }
        }

        if (ImGui::CollapsingHeader("Exemple"))
        {
            example();
        }

        ImGui::End();

    }

    void add_manager_ui(UI_Component* manager_ui) {
        _managers_ui.push_back(manager_ui);
    }

    void add_component_ui(UI_Component* component_ui) {
        _components_ui.push_back(component_ui);
    }

    void add_core_ui(UI_Display* core_ui) {
        _core_ui.push_back(core_ui);
    }

    void example() {
        ImGui::SeparatorText("General");

        static int clicked = 0;
        if (ImGui::Button("Button"))
            clicked++;
        if (clicked & 1)
        {
            ImGui::SameLine();
            ImGui::Text("Thanks for clicking me!");
        }

        static bool check = true;
        ImGui::Checkbox("checkbox", &check);

        static int e = 0;
        ImGui::RadioButton("radio a", &e, 0); ImGui::SameLine();
        ImGui::RadioButton("radio b", &e, 1); ImGui::SameLine();
        ImGui::RadioButton("radio c", &e, 2);

        // Color buttons, demonstrate using PushID() to add unique identifier in the ID stack, and changing style.
        for (int i = 0; i < 7; i++)
        {
            if (i > 0)
                ImGui::SameLine();
            ImGui::PushID(i);
            ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(i / 7.0f, 0.6f, 0.6f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(i / 7.0f, 0.7f, 0.7f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(i / 7.0f, 0.8f, 0.8f));
            ImGui::Button("Click");
            ImGui::PopStyleColor(3);
            ImGui::PopID();
        }

        // Use AlignTextToFramePadding() to align text baseline to the baseline of framed widgets elements
        // (otherwise a Text+SameLine+Button sequence will have the text a little too high by default!)
        // See 'Demo->Layout->Text Baseline Alignment' for details.
        ImGui::AlignTextToFramePadding();
        ImGui::Text("Hold to repeat:");
        ImGui::SameLine();

        // Arrow buttons with Repeater
        static int counter = 0;
        float spacing = ImGui::GetStyle().ItemInnerSpacing.x;
        ImGui::PushButtonRepeat(true);
        if (ImGui::ArrowButton("##left", ImGuiDir_Left)) { counter--; }
        ImGui::SameLine(0.0f, spacing);
        if (ImGui::ArrowButton("##right", ImGuiDir_Right)) { counter++; }
        ImGui::PopButtonRepeat();
        ImGui::SameLine();
        ImGui::Text("%d", counter);

        ImGui::Button("Tooltip");
        ImGui::SetItemTooltip("I am a tooltip");

        ImGui::LabelText("label", "Value");

        ImGui::SeparatorText("Inputs");

        {
            // To wire InputText() with std::string or any other custom string type,
            // see the "Text Input > Resize Callback" section of this demo, and the misc/cpp/imgui_stdlib.h file.
            static char str0[128] = "Hello, world!";
            ImGui::InputText("input text", str0, IM_ARRAYSIZE(str0));
            ImGui::SameLine();

            static char str1[128] = "";
            ImGui::InputTextWithHint("input text (w/ hint)", "enter text here", str1, IM_ARRAYSIZE(str1));

            static int i0 = 123;
            ImGui::InputInt("input int", &i0);

            static float f0 = 0.001f;
            ImGui::InputFloat("input float", &f0, 0.01f, 1.0f, "%.3f");

            static double d0 = 999999.00000001;
            ImGui::InputDouble("input double", &d0, 0.01f, 1.0f, "%.8f");

            static float f1 = 1.e10f;
            ImGui::InputFloat("input scientific", &f1, 0.0f, 0.0f, "%e");
            ImGui::SameLine();

            static float vec4a[4] = { 0.10f, 0.20f, 0.30f, 0.44f };
            ImGui::InputFloat3("input float3", vec4a);
        }

        ImGui::SeparatorText("Drags");

        {
            static int i1 = 50, i2 = 42;
            ImGui::DragInt("drag int", &i1, 1);
            ImGui::SameLine();

            ImGui::DragInt("drag int 0..100", &i2, 1, 0, 100, "%d%%", ImGuiSliderFlags_AlwaysClamp);

            static float f1 = 1.00f, f2 = 0.0067f;
            ImGui::DragFloat("drag float", &f1, 0.005f);
            ImGui::DragFloat("drag small float", &f2, 0.0001f, 0.0f, 0.0f, "%.06f ns");
        }

        ImGui::SeparatorText("Sliders");

        {
            static int i1 = 0;
            ImGui::SliderInt("slider int", &i1, -1, 3);
            ImGui::SameLine();

            static float f1 = 0.123f, f2 = 0.0f;
            ImGui::SliderFloat("slider float", &f1, 0.0f, 1.0f, "ratio = %.3f");
            ImGui::SliderFloat("slider float (log)", &f2, -10.0f, 10.0f, "%.4f", ImGuiSliderFlags_Logarithmic);

            static float angle = 0.0f;
            ImGui::SliderAngle("slider angle", &angle);

            // Using the format string to display a name instead of an integer.
            // Here we completely omit '%d' from the format string, so it'll only display a name.
            // This technique can also be used with DragInt().
            enum Element { Element_Fire, Element_Earth, Element_Air, Element_Water, Element_COUNT };
            static int elem = Element_Fire;
            const char* elems_names[Element_COUNT] = { "Fire", "Earth", "Air", "Water" };
            const char* elem_name = (elem >= 0 && elem < Element_COUNT) ? elems_names[elem] : "Unknown";
            ImGui::SliderInt("slider enum", &elem, 0, Element_COUNT - 1, elem_name); // Use ImGuiSliderFlags_NoInput flag to disable CTRL+Click here.
            ImGui::SameLine();
        }

        ImGui::SeparatorText("Selectors/Pickers");

        {
            static float col1[3] = { 1.0f, 0.0f, 0.2f };
            static float col2[4] = { 0.4f, 0.7f, 0.0f, 0.5f };
            ImGui::ColorEdit3("color 1", col1);
            ImGui::SameLine();

            ImGui::ColorEdit4("color 2", col2);
        }

        {
            // Using the _simplified_ one-liner Combo() api here
            // See "Combo" section for examples of how to use the more flexible BeginCombo()/EndCombo() api.
            const char* items[] = { "AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF", "GGGG", "HHHH", "IIIIIII", "JJJJ", "KKKKKKK" };
            static int item_current = 0;
            ImGui::Combo("combo", &item_current, items, IM_ARRAYSIZE(items));
            ImGui::SameLine();
        }

        {
            // Using the _simplified_ one-liner ListBox() api here
            // See "List boxes" section for examples of how to use the more flexible BeginListBox()/EndListBox() api.
            const char* items[] = { "Apple", "Banana", "Cherry", "Kiwi", "Mango", "Orange", "Pineapple", "Strawberry", "Watermelon" };
            static int item_current = 1;
            ImGui::ListBox("listbox", &item_current, items, IM_ARRAYSIZE(items), 4);
            ImGui::SameLine();
        }
    }

    void clear() {
        for (UI_Component* ui : _managers_ui) delete ui;
        for (UI_Component* ui : _components_ui) delete ui;
        for (UI_Display* ui : _core_ui) delete ui;
        _managers_ui.clear();
        _components_ui.clear();
        _core_ui.clear();
    }

    virtual ~UI_Editor() {
        clear();
    }

protected:
    void stop() {
        for (Entity* entity : Engine::GetEntities()) {
            if (entity->id() == 0) {
                TimeManager* time_manager = entity->getComponent<TimeManager>();
                if (time_manager) time_manager->disable();

                DebugManager* debug_manager = entity->getComponent<DebugManager>();
                if (debug_manager) debug_manager->pause();
            }
            entity->disable();
        }
    }

    void play() {

        for (Entity* entity : Engine::GetEntities()) {
            if (entity->id() == 0) {
                TimeManager* time_manager = entity->getComponent<TimeManager>();
                if (time_manager) time_manager->enable();

                DebugManager* debug_manager = entity->getComponent<DebugManager>();
                if (debug_manager) debug_manager->play();
            }
            entity->enable();
        }
    }

protected:
    std::vector<UI_Display*> _core_ui;
    std::vector<UI_Component*> _managers_ui;
    std::vector<UI_Component*> _components_ui;
    bool _step;
    int _nb_step;
    int _count_step;
};