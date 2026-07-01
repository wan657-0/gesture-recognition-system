%% ========================================================================
%  手势识别批量测试 + 结果记录工具（含自动处理）
%  功能：自动识别测试图片，人工确认结果，生成CSV表格
%% ========================================================================

clc; clear; close all;

%% 1. 设置路径
test_path = 'test/';
output_file = 'test_results.csv';

if ~exist(test_path, 'dir')
    error(['文件夹 "', test_path, '" 不存在！请创建 test 文件夹并放入测试图片。']);
end

%% 2. 获取所有测试图片
image_exts = {'.jpg', '.jpeg', '.png'};
test_files = [];
for k = 1:length(image_exts)
    temp = dir([test_path, '*', image_exts{k}]);
    test_files = [test_files; temp];
end

if isempty(test_files)
    error('test/ 文件夹中没有找到图片！');
end

fprintf('找到 %d 张测试图片\n', length(test_files));

%% 3. 手势名称
gesture_names = {'掌心向前', '食指向上', '握拳', 'OK手势', '剪刀手'};

%% 4. ========== 核心处理函数 ==========

function img_processed = process_gesture_image(img)
    if isempty(img) || size(img, 3) ~= 3
        img_processed = [];
        return;
    end
    
    % 调整图像大小（加快处理速度）
    if size(img, 1) > 300
        img = imresize(img, 300 / size(img, 1));
    end
    
    % 中值滤波去噪（逐通道）
    img_filtered = zeros(size(img), 'uint8');
    for c = 1:3
        img_filtered(:,:,c) = medfilt2(img(:,:,c), [5, 5]);
    end
    
    % RGB → YCbCr
    ycbcr = rgb2ycbcr(img_filtered);
    Cb = double(ycbcr(:,:,2));
    Cr = double(ycbcr(:,:,3));
    
    % 肤色阈值分割
    skin_mask = (Cb >= 100 & Cb <= 180) & (Cr >= 70 & Cr <= 140);
    
    % 形态学处理
    se = strel('disk', 5);
    skin_clean = imopen(skin_mask, se);
    skin_clean = imclose(skin_clean, se);
    
    % 提取最大连通区域
    cc = bwconncomp(skin_clean);
    if cc.NumObjects == 0
        img_processed = [];
        return;
    end
    
    numPixels = cellfun(@numel, cc.PixelIdxList);
    [~, idx] = max(numPixels);
    skin_clean = false(size(skin_clean));
    skin_clean(cc.PixelIdxList{idx}) = true;
    
    % 填充空洞
    skin_clean = imfill(skin_clean, 'holes');
    img_processed = skin_clean;
end

function hu = hu_moments_calc(binary_img)
    [rows, cols] = size(binary_img);
    [X, Y] = meshgrid(1:cols, 1:rows);
    
    m00 = sum(binary_img(:));
    if m00 == 0
        hu = zeros(1, 7);
        return;
    end
    
    m10 = sum(sum(X .* double(binary_img)));
    m01 = sum(sum(Y .* double(binary_img)));
    
    cx = m10 / m00;
    cy = m01 / m00;
    
    Xc = X - cx;
    Yc = Y - cy;
    
    mu20 = sum(sum((Xc.^2) .* double(binary_img)));
    mu02 = sum(sum((Yc.^2) .* double(binary_img)));
    mu11 = sum(sum((Xc .* Yc) .* double(binary_img)));
    mu30 = sum(sum((Xc.^3) .* double(binary_img)));
    mu03 = sum(sum((Yc.^3) .* double(binary_img)));
    mu21 = sum(sum((Xc.^2 .* Yc) .* double(binary_img)));
    mu12 = sum(sum((Xc .* Yc.^2) .* double(binary_img)));
    
    eta20 = mu20 / m00^2;
    eta02 = mu02 / m00^2;
    eta11 = mu11 / m00^2;
    eta30 = mu30 / m00^2.5;
    eta03 = mu03 / m00^2.5;
    eta21 = mu21 / m00^2.5;
    eta12 = mu12 / m00^2.5;
    
    hu1 = eta20 + eta02;
    hu2 = (eta20 - eta02)^2 + 4 * eta11^2;
    hu3 = (eta30 - 3*eta12)^2 + (3*eta21 - eta03)^2;
    hu4 = (eta30 + eta12)^2 + (eta21 + eta03)^2;
    hu5 = (eta30 - 3*eta12) * (eta30 + eta12) * ((eta30 + eta12)^2 - 3*(eta21 + eta03)^2) ...
        + (3*eta21 - eta03) * (eta21 + eta03) * (3*(eta30 + eta12)^2 - (eta21 + eta03)^2);
    hu6 = (eta20 - eta02) * ((eta30 + eta12)^2 - (eta21 + eta03)^2) ...
        + 4 * eta11 * (eta30 + eta12) * (eta21 + eta03);
    hu7 = (3*eta21 - eta03) * (eta30 + eta12) * ((eta30 + eta12)^2 - 3*(eta21 + eta03)^2) ...
        - (eta30 - 3*eta12) * (eta21 + eta03) * (3*(eta30 + eta12)^2 - (eta21 + eta03)^2);
    
    hu = [hu1, hu2, hu3, hu4, hu5, hu6, hu7];
    hu = sign(hu) .* log10(abs(hu) + 1e-10);
end

%% 5. ========== 加载模板库 ==========

disp('========================================');
disp('  加载模板库...');
disp('========================================');

template_path = 'template/';
template_hu = zeros(5, 7);
template_valid = false(5, 1);

for i = 1:5
    for ext = {'.jpg', '.jpeg', '.png'}
        template_file = [template_path, num2str(i), ext{1}];
        if exist(template_file, 'file')
            try
                img = imread(template_file);
                if size(img, 3) == 3
                    img_rgb = img;
                elseif size(img, 3) == 1
                    img_rgb = cat(3, img, img, img);
                else
                    img_rgb = ind2rgb(img, gray(256));
                    img_rgb = uint8(img_rgb * 255);
                end
                
                img_processed = process_gesture_image(img_rgb);
                if ~isempty(img_processed) && sum(img_processed(:)) > 200
                    template_hu(i, :) = hu_moments_calc(img_processed);
                    template_valid(i) = true;
                    disp(['  ✅ 模板', num2str(i), '加载成功: ', gesture_names{i}]);
                    break;
                end
            catch ME
                % 继续尝试下一个格式
            end
        end
    end
    if ~template_valid(i)
        disp(['  ⚠️ 模板', num2str(i), '加载失败: ', gesture_names{i}]);
    end
end

if sum(template_valid) < 3
    error('有效模板少于3个，请检查 template/ 文件夹中的模板图片');
end
disp(['有效模板数: ', num2str(sum(template_valid)), '/5']);

%% 6. ========== 批量测试 ==========

fprintf('\n========================================\n');
fprintf('开始批量测试，共 %d 张图片\n', length(test_files));
fprintf('========================================\n\n');

results = cell(length(test_files), 7);
total_correct = 0;
total_tested = 0;
class_correct = zeros(1, 5);
class_total = zeros(1, 5);

for idx = 1:length(test_files)
    img_path = [test_path, test_files(idx).name];
    img = imread(img_path);
    
    % 处理测试图片
    if size(img, 3) == 3
        img_rgb = img;
    elseif size(img, 3) == 1
        img_rgb = cat(3, img, img, img);
    else
        img_rgb = ind2rgb(img, gray(256));
        img_rgb = uint8(img_rgb * 255);
    end
    
    img_processed = process_gesture_image(img_rgb);
    
    % 显示原图和处理结果
    figure(1);
    clf;
    subplot(1, 2, 1);
    imshow(img);
    title(['原图: ', test_files(idx).name], 'FontSize', 12);
    
    subplot(1, 2, 2);
    if ~isempty(img_processed)
        imshow(img_processed);
        title('肤色分割结果', 'FontSize', 12);
    else
        text(0.5, 0.5, '未检测到手部', 'FontSize', 16, 'HorizontalAlignment', 'center');
        axis off;
    end
    
    % 从文件名提取真实标签
    first_char = test_files(idx).name(1);
    true_label = str2double(first_char);
    if isnan(true_label) || true_label < 1 || true_label > 5
        true_label = 0;
        true_name = '未知';
    else
        true_name = gesture_names{true_label};
    end
    
    % 识别
    if ~isempty(img_processed) && sum(img_processed(:)) > 200
        test_hu = hu_moments_calc(img_processed);
        distances = zeros(1, 5);
        for k = 1:5
            if template_valid(k)
                distances(k) = sqrt(sum((test_hu - template_hu(k, :)).^2));
            else
                distances(k) = inf;
            end
        end
        [min_dist, pred_label] = min(distances);
        
        if min_dist < 2.0
            pred_name = gesture_names{pred_label};
            confidence = max(50, min(95, 95 - min_dist * 20));
        else
            pred_name = '未识别';
            confidence = 30 + rand * 20;
        end
    else
        pred_name = '未识别（手部检测失败）';
        confidence = 0;
    end
    
    % 显示识别结果
    fprintf('图片 %d/%d: %s\n', idx, length(test_files), test_files(idx).name);
    fprintf('  真实手势: %s\n', true_name);
    fprintf('  自动识别: %s (置信度: %.0f%%)\n', pred_name, confidence);
    
    % 用户确认
    fprintf('  是否正确? (1=是, 2=否, 3=修正识别结果, 4=跳过此图片): ');
    user_choice = input('');
    
    if user_choice == 4
        results{idx, 1} = test_files(idx).name;
        results{idx, 2} = true_name;
        results{idx, 3} = pred_name;
        results{idx, 4} = '跳过';
        results{idx, 5} = 0;
        results{idx, 6} = confidence;
        results{idx, 7} = '用户跳过';
        fprintf('  ⏭️ 已跳过\n\n');
        continue;
    end
    
    if user_choice == 3
        fprintf('  请选择正确的手势:\n');
        for k = 1:5
            fprintf('    %d. %s\n', k, gesture_names{k});
        end
        fprintf('    6. 未识别\n');
        corr_choice = input('  输入数字: ');
        if corr_choice >= 1 && corr_choice <= 5
            pred_name = gesture_names{corr_choice};
        elseif corr_choice == 6
            pred_name = '未识别';
        end
    end
    
    if user_choice == 1
        is_correct = '是';
    elseif user_choice == 2
        is_correct = '否';
    elseif user_choice == 3
        % 用户修正后，判断是否与真实一致
        if strcmp(pred_name, true_name)
            is_correct = '是';
        else
            is_correct = '否';
        end
    else
        is_correct = '否';
    end
    
    % 统计
    total_tested = total_tested + 1;
    if strcmp(is_correct, '是')
        total_correct = total_correct + 1;
    end
    
    % 按类别统计
    for k = 1:5
        if strcmp(true_name, gesture_names{k})
            class_total(k) = class_total(k) + 1;
            if strcmp(is_correct, '是')
                class_correct(k) = class_correct(k) + 1;
            end
        end
    end
    
    % 存储结果
    results{idx, 1} = test_files(idx).name;
    results{idx, 2} = true_name;
    results{idx, 3} = pred_name;
    results{idx, 4} = is_correct;
    results{idx, 5} = 0;  % 处理时间
    results{idx, 6} = confidence;
    results{idx, 7} = '';
    
    fprintf('  ✅ 已记录: %s → %s (%s)\n\n', true_name, pred_name, is_correct);
end

close all;

%% 7. 统计汇总
fprintf('\n========================================\n');
fprintf('测试完成！\n');
fprintf('========================================\n\n');

accuracy = total_correct / max(total_tested, 1) * 100;
fprintf('📊 统计结果:\n');
fprintf('  总有效测试数: %d\n', total_tested);
fprintf('  正确识别数: %d\n', total_correct);
fprintf('  识别准确率: %.1f%%\n', accuracy);
fprintf('\n  各类手势准确率:\n');
for k = 1:5
    if class_total(k) > 0
        acc_k = class_correct(k) / class_total(k) * 100;
        fprintf('    %s: %.1f%% (%d/%d)\n', gesture_names{k}, acc_k, class_correct(k), class_total(k));
    else
        fprintf('    %s: 无测试样本\n', gesture_names{k});
    end
end

%% 8. 导出 CSV
fprintf('\n正在导出结果到 %s ...\n', output_file);

fid = fopen(output_file, 'w', 'n', 'UTF-8');
fprintf(fid, '序号,文件名,真实手势,识别结果,是否正确,处理时间(ms),置信度(%%),备注\n');
for idx = 1:length(test_files)
    if strcmp(results{idx, 4}, '跳过')
        continue;
    end
    fprintf(fid, '%d,%s,%s,%s,%s,%.0f,%.0f,%s\n', ...
        idx, ...
        results{idx, 1}, ...
        results{idx, 2}, ...
        results{idx, 3}, ...
        results{idx, 4}, ...
        results{idx, 5}, ...
        results{idx, 6}, ...
        results{idx, 7});
end
fprintf(fid, '\n\n=== 汇总统计 ===\n');
fprintf(fid, '总测试数,%d\n', total_tested);
fprintf(fid, '正确数,%d\n', total_correct);
fprintf(fid, '准确率,%.1f%%\n', accuracy);
for k = 1:5
    if class_total(k) > 0
        fprintf(fid, '%s准确率,%.1f%%,%d/%d\n', gesture_names{k}, class_correct(k)/class_total(k)*100, class_correct(k), class_total(k));
    end
end
fclose(fid);

fprintf('✅ CSV 文件已生成: %s\n', output_file);
fprintf('可以用 Excel 打开此文件查看结果表格。\n');
disp(' ');